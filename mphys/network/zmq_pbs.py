import argparse
import atexit
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback

import zmq
from pbs4py import PBS
from pbs4py.job import PBSJob

from mphys.network import RemoteComp, RemoteComponentError, Server, ServerManager


# MPhysZeroMQServer error replies start with this; "status" must be the first key
SERVER_ERROR_PREFIX = b'{"status": "error"'


def _pbs_command_env():
    """
    Environment for running PBS client commands. When the client itself runs
    in a multi-node PBS job, $TMPDIR points to a per-job directory
    (/var/tmp/pbs.<jobid>) that only exists on the job's first node; qsub
    fails with "could not create/open tmp file" on the other nodes. Fall back
    to a temp directory that exists in that case.
    """
    env = dict(os.environ)
    tmpdir = env.get("TMPDIR")
    if tmpdir and not os.path.isdir(tmpdir):
        for fallback in ("/var/tmp", "/tmp"):
            if os.path.isdir(fallback):
                env["TMPDIR"] = fallback
                break
        else:
            env.pop("TMPDIR")
    return env


class PBSJobWithTimeout(PBSJob):
    """
    pbs4py PBSJob whose qstat call is given a timeout, so that a PBS server
    that is down (PBS clients retry the connection for a long time) does not
    block the client indefinitely. A timeout raises subprocess.TimeoutExpired,
    which the server manager treats like any other failed qstat.
    """

    qstat_timeout = 120  # seconds

    def _run_qstat_to_get_full_job_attributes(self):
        result = subprocess.run(
            ["qstat", "-xf", str(self.id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.qstat_timeout,
            env=_pbs_command_env(),
        )
        return result.stdout.decode("utf-8", errors="replace").split("\n")


def _terminate_when_parent_dies():
    """
    Runs in the child between fork and exec. On Linux, ask the kernel to send
    SIGTERM to this process when its parent thread exits, so the ssh tunnel
    does not outlive the client even if the client is killed abruptly.
    """
    if sys.platform.startswith("linux"):
        PR_SET_PDEATHSIG = 1
        try:
            ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except (AttributeError, OSError):
            pass


def _exit_on_signal(signum, frame):
    # os.write rather than print: the signal may arrive while the main thread
    # is inside a buffered stdout write, and re-entering that raises RuntimeError
    try:
        os.write(
            sys.stdout.fileno(),
            f"CLIENT: Received signal {signal.Signals(signum).name}; shutting down remote servers\n".encode(),
        )
    except (OSError, ValueError, AttributeError):
        pass
    raise SystemExit(128 + signum)


def _install_shutdown_signal_handlers():
    """
    Convert SIGTERM/SIGHUP into SystemExit so atexit handlers (and thus
    stop_server) run. Replaces the default handler and C-level handlers
    (signal.getsignal returns None for those, e.g. ones installed by an MPI
    runtime such as MPT, which would otherwise just exit without cleanup).
    Python-level user handlers and SIG_IGN are left untouched. No-op if not
    called from the main thread.
    """
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            if signal.getsignal(sig) in (signal.SIG_DFL, None):
                signal.signal(sig, _exit_on_signal)
        except (ValueError, OSError, AttributeError):
            pass


class RemoteZeroMQComp(RemoteComp):
    """
    A derived RemoteComp class that uses pbs4py for HPC job management
    and ZeroMQ for network communication.
    """

    def initialize(self):
        self.options.declare("pbs", "pbs4py Launcher object")
        self.options.declare(
            "port", default=5081, desc="port number for server/client communication"
        )
        self.options.declare(
            "acceptable_port_range",
            default=[5081, 6000],
            desc="port range to look through if 'port' is currently busy",
        )
        self.options.declare(
            "forward_through_frontend",
            default=False,
            desc="whether to have ssh port forwarding jump through frontend node, in case compute nodes cannot communicate",
        )
        self.options.declare(
            "additional_server_args",
            default="",
            desc="Optional arguments to give server, in addition to --port <port number>",
        )
        self.options.declare(
            "job_expiration_max_restarts",
            default=None,
            desc="Optional maximum number of server restarts due to job expiration; unlimited by default",
        )
        self.options.declare(
            "startup_ping_interval",
            default=10.0,
            types=(int, float),
            desc="seconds between pings sent to a newly launched server until it replies",
        )
        self.options.declare(
            "startup_timeout",
            default=None,
            desc="seconds to wait for a newly launched server to reply to pings before raising an error; unlimited by default",
        )
        self.options.declare(
            "receive_timeout",
            default=300.0,
            types=(int, float),
            desc="seconds to wait for a server reply before re-sending the request (in case it was lost) "
            + "and checking whether the server job is still running",
        )
        super().initialize()
        self.server_manager = (
            None  # for avoiding reinitialization due to multiple setup calls
        )

    def _send_inputs_to_server(self, remote_input_dict, command: str):
        if self._doing_derivative_evaluation(command):
            print(
                f"CLIENT (subsystem {self.name}): Requesting derivative call from server",
                flush=True,
            )
        else:
            print(
                f"CLIENT (subsystem {self.name}): Requesting function call from server",
                flush=True,
            )
        input_str = f"{command}|{str(json.dumps(remote_input_dict))}"
        self.server_manager.send_request(input_str.encode())

    def _receive_outputs_from_server(self):
        return json.loads(self.server_manager.receive_reply().decode())

    def _setup_server_manager(self):
        if self.server_manager is None:
            self.server_manager = MPhysZeroMQServerManager(
                pbs=self.options["pbs"],
                run_server_filename=self.options["run_server_filename"],
                component_name=self.name,
                port=self.options["port"],
                acceptable_port_range=self.options["acceptable_port_range"],
                forward_through_frontend=self.options["forward_through_frontend"],
                additional_server_args=self.options["additional_server_args"],
                job_expiration_max_restarts=self.options["job_expiration_max_restarts"],
                startup_ping_interval=self.options["startup_ping_interval"],
                startup_timeout=self.options["startup_timeout"],
                receive_timeout=self.options["receive_timeout"],
            )


class MPhysZeroMQServerManager(ServerManager):
    """
    A derived ServerManager class that uses pbs4py for HPC job management
    and ZeroMQ for network communication.

    Parameters
    ----------
    pbs : :class:`~pbs4py.PBS`
        pbs4py launcher used for HPC job management
    run_server_filename : str
        Python filename that initializes and runs the :class:`~mphys.network.zmq_pbs.MPhysZeroMQServer` server
    component_name : str
        Name of the remote component, for capturing output from separate remote components to mphys_{component_name}_server{server_number}.out
    port : int
        Desired port number for ssh port forwarding
    acceptable_port_range : list
        Range of alternative port numbers if specified port is already in use
    forward_through_frontend: bool
        Setup ssh forwarding to jump through frontend node ($PBS_O_HOST). For cases where compute nodes cannot communicate
    additional_server_args : str
        Optional arguments to give server, in addition to --port <port number>
    job_expiration_max_restarts : int
        Optional maximum number of server restarts due to job expiration; unlimited by default
    startup_ping_interval : float
        Seconds between pings sent to a newly launched server until it replies
    startup_timeout : float
        Seconds to wait for a newly launched server to reply to pings before raising an error; unlimited if None
    receive_timeout : float
        Seconds to wait for a server reply before re-sending the request (in case it was lost in transit)
        and checking whether the server job is still running
    """

    def __init__(
        self,
        pbs: PBS,
        run_server_filename: str,
        component_name: str,
        port=5081,
        acceptable_port_range=[5081, 6000],
        forward_through_frontend=False,
        additional_server_args="",
        job_expiration_max_restarts=None,
        startup_ping_interval=10.0,
        startup_timeout=None,
        receive_timeout=300.0,
    ):
        self.pbs = pbs
        self.run_server_filename = run_server_filename
        self.component_name = component_name
        self.port = port
        self.acceptable_port_range = acceptable_port_range
        self.forward_through_frontend = forward_through_frontend
        self.additional_server_args = additional_server_args
        self.job_expiration_max_restarts = job_expiration_max_restarts
        self.startup_ping_interval = startup_ping_interval
        self.startup_timeout = startup_timeout
        self.receive_timeout = receive_timeout
        self._server_output_file = None
        self.queue_time_delay = (
            5  # seconds to wait before rechecking if a job has started
        )
        self.job_check_interval = (
            60  # seconds between qstat calls while waiting on the server
        )
        self.job_gone_checks_required = (
            3  # consecutive non-running qstat results before declaring the job dead
        )
        # retries for PBS commands (qsub/qstat/qdel) failing, e.g. when the PBS server is unreachable
        self.pbs_retry_attempts = 10
        self.pbs_retry_delay = 60  # seconds
        self.pbs_command_timeout = (
            120  # seconds before a hung qsub/qstat/qdel is abandoned
        )
        self.qdel_retry_attempts = 3
        self.qdel_retry_delay = 10  # seconds
        self.server_counter = 0  # for saving output of each server to different files
        self.job_expiration_restarts = 0
        self.shutdown_send_timeout_ms = 5000
        self.server_stopped = True
        self.socket = None
        self._pending_request = None
        _install_shutdown_signal_handlers()
        atexit.register(self._stop_server_at_exit)
        self.start_server()

    def start_server(self):
        self._initialize_connection()
        self.server_counter += 1
        self._launch_job()
        self.server_stopped = False
        self._wait_for_server_to_be_ready()

    def send_request(self, message: bytes):
        """
        Send a request to the server. The message is kept so that it can be
        re-sent by receive_reply if the reply does not arrive in time.
        """
        self._pending_request = message
        self.socket.send(message)

    def receive_reply(self) -> bytes:
        """
        Wait for the server's reply to the last request. If no reply arrives
        within receive_timeout seconds, the request is re-sent in case it (or
        the reply) was lost in transit; the server skips re-evaluating a design
        it has already evaluated, so this is safe. If the server's job is found
        to have ended, the server is restarted and the request re-sent.
        """
        job_gone_count = 0
        while True:
            if self.socket.poll(int(self.receive_timeout * 1000)):
                reply = self.socket.recv()
                self._raise_if_server_error(reply)
                return reply

            running = self._job_is_running()
            if (
                running or running is None
            ):  # None: PBS could not be queried; assume still running
                if running:
                    job_gone_count = 0
                print(
                    f"CLIENT (subsystem {self.component_name}): No reply from server after {self.receive_timeout} s; "
                    + "re-sending request in case it was lost",
                    flush=True,
                )
                self._reset_zmq_socket()
            else:
                job_gone_count += 1
                if job_gone_count < self.job_gone_checks_required:
                    print(
                        f"CLIENT (subsystem {self.component_name}): No reply from server and qstat does not report a "
                        + f"running job ({job_gone_count}/{self.job_gone_checks_required} checks)",
                        flush=True,
                    )
                    continue
                job_gone_count = 0
                print(
                    f"CLIENT (subsystem {self.component_name}): Server job ended while waiting for a reply; "
                    + "restarting server and re-sending request",
                    flush=True,
                )
                self._count_job_expiration_restart()
                self.stop_server()
                self.start_server()
            self.socket.send(self._pending_request)

    def _job_is_running(self):
        """
        Returns True/False, or None if the job state could not be determined
        (e.g. the PBS server is unreachable).
        """
        state = self._query_job_state()
        if state is None:
            return None
        return state == "R"

    def _query_job_state(self):
        """
        Refresh the job's attributes from qstat, retrying if PBS cannot be
        queried (unreachable server, garbled output). Returns the job state
        string, or None if PBS could not be queried after all retries.
        """
        for attempt in range(1, self.pbs_retry_attempts + 1):
            try:
                self.job.update_job_state()
                return self.job.state
            except Exception as e:
                print(
                    f"CLIENT (subsystem {self.component_name}): qstat failed ({e!r}); "
                    + f"retrying in {self.pbs_retry_delay} s ({attempt}/{self.pbs_retry_attempts})",
                    flush=True,
                )
                time.sleep(self.pbs_retry_delay)
        print(
            f"CLIENT (subsystem {self.component_name}): Could not query PBS for job {self.job.id}; "
            + "treating the job state as unknown",
            flush=True,
        )
        return None

    def _count_job_expiration_restart(self):
        if self.job_expiration_max_restarts is not None:
            if self.job_expiration_restarts + 1 > self.job_expiration_max_restarts:
                self.stop_server()
                raise RemoteComponentError(
                    f"CLIENT (subsystem {self.component_name}): Reached maximum number of job expiration restarts"
                )
            self.job_expiration_restarts += 1

    def _wait_for_server_to_be_ready(self):
        """
        Ping the server until it replies. The PBS job reports as running long
        before the server has set up its model and bound its port; a request
        sent through the ssh tunnel before then is silently dropped, which
        would deadlock the REQ/REP pair.
        """
        print(
            f"CLIENT (subsystem {self.component_name}): Waiting for server to be ready",
            flush=True,
        )
        start_time = time.time()
        last_job_check = start_time
        job_gone_count = 0
        ping_timeout_ms = int(self.startup_ping_interval * 1000)
        while True:
            self.socket.send(b"ping|null")
            if self.socket.poll(ping_timeout_ms):
                reply = self.socket.recv()
                self._raise_if_server_error(reply)
                if json.loads(reply.decode()) == Server.PING_REPLY:
                    break
                print(
                    f"CLIENT (subsystem {self.component_name}): Unexpected reply to ping: {reply[:80]!r}",
                    flush=True,
                )
            self._reset_zmq_socket()

            if (
                self.startup_timeout is not None
                and time.time() - start_time > self.startup_timeout
            ):
                raise RemoteComponentError(
                    f"CLIENT (subsystem {self.component_name}): Server did not become ready within "
                    + f"{self.startup_timeout} s"
                )
            if time.time() - last_job_check > self.job_check_interval:
                last_job_check = time.time()
                running = self._job_is_running()
                if running:
                    job_gone_count = 0
                elif running is False:  # None (PBS unreachable) is inconclusive
                    job_gone_count += 1
                    if job_gone_count >= self.job_gone_checks_required:
                        raise RemoteComponentError(
                            f"CLIENT (subsystem {self.component_name}): Server job {self.job.id} ended before the "
                            + "server became ready"
                            + self._server_output_tail()
                        )
        print(
            f"CLIENT (subsystem {self.component_name}): Server is ready "
            + f"(startup time: {time.time() - start_time:.1f} s)",
            flush=True,
        )

    def _raise_if_server_error(self, reply: bytes):
        """
        Raise if the reply is an error report from MPhysZeroMQServer: the
        server failed, and retrying or relaunching it would fail the same way.
        """
        if not reply.startswith(SERVER_ERROR_PREFIX):
            return
        try:
            info = json.loads(reply.decode())
        except ValueError:
            info = {"traceback": reply.decode(errors="replace")}
        location = f" on {info.get('host')}" if info.get("host") else ""
        if info.get("rank") is not None:
            location += f" (rank {info['rank']})"
        try:
            self.stop_server()
        except Exception as e:
            print(
                f"CLIENT (subsystem {self.component_name}): Error stopping failed server: {e!r}",
                flush=True,
            )
        raise RemoteComponentError(
            f"CLIENT (subsystem {self.component_name}): The remote analysis server failed{location} "
            + f"with the following error:\n{info.get('traceback', '').rstrip()}"
        )

    def _server_output_tail(self, num_lines=30) -> str:
        path = self._server_output_file
        if not path or not os.path.isfile(path):
            return "; check the server output file for errors"
        try:
            with open(path, errors="replace") as f:
                lines = f.readlines()[-num_lines:]
        except OSError:
            return f"; check {path} for errors"
        return f". Last lines of {path}:\n" + "".join(lines).rstrip()

    def _reset_zmq_socket(self):
        # a REQ socket that has sent without receiving cannot send again;
        # drop it (and any unsent message) and connect a fresh one
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self._initialize_zmq_socket()

    def stop_server(self):
        if not self.server_stopped:
            print(
                f"CLIENT (subsystem {self.component_name}): Stopping the remote analysis server",
                flush=True,
            )
            self.server_stopped = True
            try:
                if self.job.state == "R":
                    self.socket.setsockopt(zmq.SNDTIMEO, self.shutdown_send_timeout_ms)
                    self.socket.send("shutdown|null".encode())
            except Exception as e:
                print(
                    f"CLIENT (subsystem {self.component_name}): Could not send shutdown message to server ({e!r}); deleting job directly",
                    flush=True,
                )
            self._shutdown_server()
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()

    def _stop_server_at_exit(self):
        if not self.server_stopped:
            print(
                f"CLIENT (subsystem {self.component_name}): Python exiting; stopping the analysis server",
                flush=True,
            )
            try:
                self.stop_server()
            except Exception as e:
                print(
                    f"CLIENT (subsystem {self.component_name}): Error while stopping server at exit: {e!r}",
                    flush=True,
                )

    def enough_time_is_remaining(self, estimated_model_time):
        if self._query_job_state() is None:
            print(
                f"CLIENT (subsystem {self.component_name}): Cannot determine remaining walltime; "
                + "assuming enough time remains",
                flush=True,
            )
            return True
        if self.job.walltime_remaining is None:
            return False
        else:
            return estimated_model_time < self.job.walltime_remaining

    def job_has_expired(self):
        state = self._query_job_state()
        if state is None:
            print(
                f"CLIENT (subsystem {self.component_name}): Cannot determine job state; "
                + "assuming the server is still running",
                flush=True,
            )
            return False
        if state == "R" and not self.server_stopped:
            return False
        else:
            self._count_job_expiration_restart()
            print(
                f"CLIENT (subsystem {self.component_name}): Job no longer running; flagging for job restart"
            )
            return True

    def _port_is_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _initialize_connection(self):
        if self._port_is_in_use(self.port):
            print(
                f"CLIENT (subsystem {self.component_name}): Port {self.port} is already in use... finding first available port in the range {self.acceptable_port_range}",
                flush=True,
            )

            for port in range(
                self.acceptable_port_range[0], self.acceptable_port_range[1] + 1
            ):
                if not self._port_is_in_use(port):
                    self.port = port
                    break
            else:
                raise RemoteComponentError(
                    f"CLIENT (subsystem {self.component_name}): Could not find open port"
                )

        self._initialize_zmq_socket()

    def _initialize_zmq_socket(self):
        # shared context: sockets are recreated on every retry, and each
        # zmq.Context() would otherwise leak an IO thread
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")

    def _launch_job(self):
        print(
            f"CLIENT (subsystem {self.component_name}): Launching new server",
            flush=True,
        )
        python_command = f"python {self.run_server_filename} --port {self.port} {self.additional_server_args}"
        output_root_name = f"mphys_{self.component_name}_server{self.server_counter}"
        self._server_output_file = os.path.abspath(f"{output_root_name}.out")
        python_mpi_command = self.pbs.create_mpi_command(
            python_command,
            output_root_name=output_root_name,
        )
        jobid = self._submit_job(f"MPhys{self.port}", [python_mpi_command])
        self.job = self._create_job_handle(jobid)
        self._wait_for_job_to_start()
        self._setup_ssh()

    def _create_job_handle(self, jobid) -> PBSJob:
        # PBSJob's constructor queries qstat, which may fail if PBS is unreachable
        PBSJobWithTimeout.qstat_timeout = self.pbs_command_timeout
        for attempt in range(1, self.pbs_retry_attempts + 1):
            try:
                return PBSJobWithTimeout(jobid)
            except Exception as e:
                print(
                    f"CLIENT (subsystem {self.component_name}): qstat failed for new job {jobid} ({e!r}); "
                    + f"retrying in {self.pbs_retry_delay} s ({attempt}/{self.pbs_retry_attempts})",
                    flush=True,
                )
                time.sleep(self.pbs_retry_delay)
        raise RemoteComponentError(
            f"CLIENT (subsystem {self.component_name}): Could not query PBS for new job {jobid}"
        )

    # qsub stderr fragments indicating a problem that may resolve itself if we retry
    TRANSIENT_QSUB_ERRORS = (
        "cannot connect",
        "Connection refused",
        "Connection reset",
        "Connection timed out",
        "timed out",
        "No route to host",
        "Communication failure",
        "Temporarily unavailable",
        "Resource temporarily unavailable",
        "server is busy",
        "Server shutting down",
        "Unknown Host",
    )

    def _submit_job(self, job_name, job_body) -> str:
        """
        qsub the job, retrying if PBS appears to be temporarily unreachable.
        Any other qsub error (e.g. "Unauthorized Request", a bad queue name)
        is raised immediately with qsub's message.
        """
        for attempt in range(1, self.pbs_retry_attempts + 1):
            jobid, error, retryable = self._run_qsub(job_name, job_body)
            if self._is_valid_job_id(jobid):
                return jobid.strip()
            reason = error or f"qsub returned {jobid!r}"
            if not retryable:
                raise RemoteComponentError(
                    f"CLIENT (subsystem {self.component_name}): Could not submit server job: {error}"
                )
            print(
                f"CLIENT (subsystem {self.component_name}): Job submission failed ({reason}); "
                + f"retrying in {self.pbs_retry_delay} s ({attempt}/{self.pbs_retry_attempts})",
                flush=True,
            )
            time.sleep(self.pbs_retry_delay)
        raise RemoteComponentError(
            f"CLIENT (subsystem {self.component_name}): Could not submit server job after "
            + f"{self.pbs_retry_attempts} attempts"
        )

    def _run_qsub(self, job_name, job_body):
        """
        Write the job script with pbs4py and run qsub ourselves so that qsub's
        stderr is captured (pbs4py's launch() only returns stdout). Returns
        (stdout, error_message, retryable). Non-PBS launchers (e.g. pbs4py's
        fake launcher for testing) fall back to launcher.launch(), with any
        exception treated as retryable since its cause is unknown.
        """
        if not isinstance(self.pbs, PBS):
            try:
                return self.pbs.launch(job_name, job_body, blocking=False), "", True
            except Exception as e:
                return None, repr(e), True
        filename = f"{job_name}.{self.pbs.batch_file_extension}"
        self.pbs.write_job_file(filename, job_name, job_body)
        try:
            result = subprocess.run(
                ["qsub", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.pbs_command_timeout,
                env=_pbs_command_env(),
            )
        except subprocess.TimeoutExpired:  # PBS server down: qsub blocks retrying its connection
            return (
                None,
                f"qsub did not return within {self.pbs_command_timeout} s",
                True,
            )
        except OSError as e:  # e.g. qsub not on PATH
            return None, repr(e), False
        stdout = result.stdout.strip()
        if stdout:
            print(stdout, flush=True)  # the job id, as pbs4py's launch() would print
        error = result.stderr.strip()
        retryable = not error or self._is_transient_qsub_error(error)
        return stdout, error, retryable

    @classmethod
    def _is_transient_qsub_error(cls, error: str) -> bool:
        error_lower = error.lower()
        return any(
            fragment.lower() in error_lower for fragment in cls.TRANSIENT_QSUB_ERRORS
        )

    @staticmethod
    def _is_valid_job_id(jobid) -> bool:
        if not isinstance(jobid, str):
            return False
        jobid = jobid.strip()
        return jobid[:1].isdigit() or "FakePBS" in jobid

    def _wait_for_job_to_start(self):
        print(
            f"CLIENT (subsystem {self.component_name}): Waiting for job to start",
            flush=True,
        )
        job_submission_time = time.time()
        self._setup_dummy_socket()
        while self.job.state != "R":
            time.sleep(self.queue_time_delay)
            if self._query_job_state() == "F":
                self._stop_dummy_socket()
                raise RemoteComponentError(
                    f"CLIENT (subsystem {self.component_name}): Server job {self.job.id} finished before it "
                    + f"started running (exit status {self.job.exit_status})"
                    + self._server_output_tail()
                )
        self._stop_dummy_socket()
        self.job_start_time = time.time()
        print(
            f"CLIENT (subsystem {self.component_name}): Job started (queue wait time: {(time.time()-job_submission_time)/3600} hours)",
            flush=True,
        )

    def _setup_ssh(self):
        front_end_host = os.environ.get("PBS_O_HOST")
        if front_end_host is not None and self.forward_through_frontend:
            ssh_command = f"ssh -4 -o ServerAliveCountMax=40 -o ServerAliveInterval=15 -N -L {self.port}:localhost:{self.port} -J {front_end_host} {self.job.hostname}"
        else:
            ssh_command = f"ssh -4 -o ServerAliveCountMax=40 -o ServerAliveInterval=15 -N -L {self.port}:localhost:{self.port} {self.job.hostname}"
        self.ssh_proc = subprocess.Popen(
            ssh_command.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=_terminate_when_parent_dies,
        )

    def _kill_ssh(self):
        if self.ssh_proc.poll() is None:
            self.ssh_proc.kill()

    def _shutdown_server(self):
        try:
            self._kill_ssh()
        finally:
            time.sleep(0.1)  # prevent full shutdown before job deletion?
            self._qdel()

    def _qdel(self) -> bool:
        """
        Delete the server job, retrying if PBS is unreachable. Returns whether
        the job is known to be gone.
        """
        jobid = str(self.job.id)
        print(f"qdel {jobid}", flush=True)
        for attempt in range(1, self.qdel_retry_attempts + 1):
            try:
                result = subprocess.run(
                    ["qdel", jobid],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.pbs_command_timeout,
                    env=_pbs_command_env(),
                )
                error = result.stderr.strip()
                if result.returncode == 0:
                    return True
                # job already gone: nothing left to do
                if "Unknown Job Id" in error or "Job has finished" in error:
                    return True
            except subprocess.TimeoutExpired:
                error = f"qdel did not return within {self.pbs_command_timeout} s"
            except OSError as e:
                error = repr(e)
            print(
                f"CLIENT (subsystem {self.component_name}): qdel failed ({error}); "
                + f"retrying in {self.qdel_retry_delay} s ({attempt}/{self.qdel_retry_attempts})",
                flush=True,
            )
            time.sleep(self.qdel_retry_delay)
        print(
            f"CLIENT (subsystem {self.component_name}): WARNING: could not delete server job {jobid}; "
            + "it may still be running and should be deleted manually",
            flush=True,
        )
        return False

    def _setup_dummy_socket(self):
        print(
            f"CLIENT (subsystem {self.component_name}): Starting dummy ZeroMQ socket to hold port {self.port} while in queue",
            flush=True,
        )
        context = zmq.Context.instance()
        self.dummy_socket = context.socket(zmq.REP)
        self.dummy_socket.bind(f"tcp://*:{self.port}")

    def _stop_dummy_socket(self):
        self.dummy_socket.setsockopt(zmq.LINGER, 0)
        self.dummy_socket.close()


class MPhysZeroMQServer(Server):
    """
    A derived Server class that uses ZeroMQ for network communication.

    Parameters
    ----------
    port : int
        TCP port to listen on
    report_errors : bool
        If the server raises during startup (model import/setup) or while
        handling a request, send the traceback to the client as the reply to
        its next message, so the client raises immediately instead of
        retrying or relaunching a server that would fail the same way.
        Errors raised before this object is created (e.g. import errors at the
        top of the server script) cannot be reported this way; the client then
        detects the ended job and shows the end of the server output file.

    Other parameters are passed to :class:`~mphys.network.Server`.
    """

    ERROR_REPORT_TIMEOUT = 300  # seconds rank 0 waits for the client to contact it
    NONROOT_ERROR_GRACE_PERIOD = (
        360  # seconds other ranks wait for rank 0 before aborting
    )

    def __init__(
        self,
        port,
        get_om_group_function_pointer,
        ignore_setup_warnings=False,
        ignore_runtime_warnings=False,
        rerun_initial_design=False,
        write_n2=False,
        report_errors=True,
    ):
        self.port = port
        self.report_errors = report_errors
        self.socket = None
        try:
            super().__init__(
                get_om_group_function_pointer,
                ignore_setup_warnings,
                ignore_runtime_warnings,
                rerun_initial_design,
                write_n2,
            )
            self._setup_zeromq_socket(port)
        except Exception:
            self._handle_server_error()

    def run(self):
        try:
            super().run()
        except Exception:
            self._handle_server_error()

    def _setup_zeromq_socket(self, port):
        if self.comm.rank == 0:
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{port}")

    def _parse_incoming_message(self):
        inputs = None
        if self.comm.rank == 0:
            inputs = self.socket.recv().decode()
        inputs = self.prob.model.comm.bcast(inputs)

        command, input_dict = inputs.split("|")
        if command != "shutdown":
            input_dict = json.loads(input_dict)
        return command, input_dict

    def _send_outputs_to_client(self, output_dict: dict):
        if self.comm.rank == 0:
            self.socket.send(str(json.dumps(output_dict)).encode())

    def _handle_server_error(self):
        """
        Called from an except block. Report the error to the client from rank
        0, then make sure the whole MPI job ends (other ranks may be blocked in
        collectives). Re-raises the original exception if nothing needs aborting.
        """
        if not self.report_errors:
            raise
        error_text = traceback.format_exc()
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        comm = getattr(self, "comm", None) or world
        print(
            f"SERVER (rank {comm.rank}): failed with the following error:\n{error_text}",
            file=sys.stderr,
            flush=True,
        )
        if comm.rank == 0:
            reply = _server_error_reply(error_text, comm.rank)
            if _report_error_to_client(
                self.socket, self.port, reply, self.ERROR_REPORT_TIMEOUT
            ):
                print("SERVER: error reported to client", flush=True)
        elif world.size > 1:
            time.sleep(self.NONROOT_ERROR_GRACE_PERIOD)
        if world.size > 1:
            world.Abort(1)
        raise


def _server_error_reply(error_text, rank=None) -> bytes:
    # "status" must be the first key: the client detects error replies by SERVER_ERROR_PREFIX
    return json.dumps(
        {
            "status": "error",
            "host": socket.gethostname(),
            "rank": rank,
            "traceback": error_text,
        }
    ).encode()


def _send_and_close(sock, reply):
    sock.send(reply)
    sock.setsockopt(zmq.LINGER, 5000)  # give the reply time to leave
    time.sleep(1.0)
    sock.close()


def _report_error_to_client(sock, port, reply, timeout) -> bool:
    """
    Send an error reply to the client. With an existing REP socket that has
    just received a request, the reply answers that request; otherwise wait
    for the client's next message (e.g. a startup ping). Returns whether sent.
    """
    if sock is not None and not sock.closed:
        try:
            _send_and_close(sock, reply)
            return True
        except zmq.ZMQError:
            pass  # not waiting to send: wait for the next request below
    else:
        sock = zmq.Context.instance().socket(zmq.REP)
        try:
            sock.bind(f"tcp://*:{port}")
        except zmq.ZMQError as e:
            print(
                f"SERVER: could not bind port {port} to report error ({e})", flush=True
            )
            sock.close(linger=0)
            return False
    if not sock.poll(int(timeout * 1000)):
        print(f"SERVER: client did not contact server within {timeout} s", flush=True)
        sock.close(linger=0)
        return False
    sock.recv()
    _send_and_close(sock, reply)
    return True


def get_default_zmq_pbs_argparser():
    parser = argparse.ArgumentParser(
        "Python script for launching mphys analysis server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, help="tcp port number for zeromq socket")
    return parser
