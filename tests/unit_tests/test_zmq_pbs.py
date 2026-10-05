import contextlib
import io
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

try:
    import zmq

    from mphys.network import zmq_pbs
except ImportError:
    zmq = None
    zmq_pbs = None

skip_without_zmq = unittest.skipIf(
    zmq_pbs is None, "zmq and pbs4py are required for mphys.network.zmq_pbs"
)


def make_manager(job_state="R", ssh_running=True, **attrs):
    """
    Build an MPhysZeroMQServerManager without running __init__ (which would
    submit a PBS job), with mocked job, socket, and ssh process.
    """
    manager = zmq_pbs.MPhysZeroMQServerManager.__new__(zmq_pbs.MPhysZeroMQServerManager)
    manager.component_name = "test"
    manager.port = 5081
    manager.acceptable_port_range = [5081, 5090]
    manager.forward_through_frontend = False
    manager.shutdown_send_timeout_ms = 100
    manager.startup_ping_interval = 0.2
    manager.startup_timeout = None
    manager.receive_timeout = 0.2
    manager.job_check_interval = 60
    manager.job_gone_checks_required = 3
    manager.job_expiration_max_restarts = None
    manager.job_expiration_restarts = 0
    manager.queue_time_delay = 0
    manager.pbs_retry_attempts = 2
    manager.pbs_retry_delay = 0
    manager.pbs_command_timeout = 120
    manager.qdel_retry_attempts = 2
    manager.qdel_retry_delay = 0
    manager.server_stopped = False
    manager._pending_request = None
    manager._server_output_file = None
    manager.pbs = mock.MagicMock()
    manager.job = mock.MagicMock()
    manager.job.state = job_state
    manager.job.hostname = "r101i0n0"
    manager.job.id = "12345.pbssrv1"
    manager.socket = mock.MagicMock()
    manager.ssh_proc = mock.MagicMock()
    manager.ssh_proc.poll.return_value = None if ssh_running else 0
    manager._qdel = mock.MagicMock(return_value=True)
    for key, val in attrs.items():
        setattr(manager, key, val)
    return manager


def qdel_result(returncode=0, stderr=""):
    return subprocess.CompletedProcess(["qdel"], returncode, stdout="", stderr=stderr)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class FakeZmqServer(threading.Thread):
    """
    Stand-in for the server end of the ssh tunnel, running in a thread.

    For the first `refuse_for` seconds it behaves like an ssh local forward
    whose remote end is not yet listening: TCP connections are accepted and
    immediately closed, so anything the client sent is lost. It then binds a
    ROUTER socket (so it can choose not to reply, unlike REP) and answers each
    message with handler(message) -> bytes, or drops it if handler returns None.
    """

    def __init__(self, port, handler, refuse_for=0.0):
        super().__init__(daemon=True)
        self.port = port
        self.handler = handler
        self.refuse_for = refuse_for
        self.received = []
        self._stop_event = threading.Event()
        self.bound = threading.Event()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def _refuse_connections(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("localhost", self.port))
        listener.listen(5)
        deadline = time.time() + self.refuse_for
        while time.time() < deadline and not self._stop_event.is_set():
            ready, _, _ = select.select([listener], [], [], 0.05)
            if ready:
                conn, _ = listener.accept()
                conn.close()
        listener.close()

    def run(self):
        if self.refuse_for > 0:
            self._refuse_connections()
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{self.port}")
        self.bound.set()
        try:
            while not self._stop_event.is_set():
                if not sock.poll(50):
                    continue
                frames = sock.recv_multipart()
                identity, message = frames[0], frames[-1]
                self.received.append(message)
                reply = self.handler(message)
                if reply is not None:
                    sock.send_multipart([identity, b"", reply])
        finally:
            sock.close()


def ping_reply():
    return json.dumps(zmq_pbs.Server.PING_REPLY).encode()


def make_live_manager(test_case, **attrs):
    """
    A manager with a real REQ socket connected to a free local port; the
    socket is closed when the test finishes.
    """
    port = free_port()
    manager = make_manager(port=port, **attrs)
    manager._initialize_zmq_socket()

    def close_socket():
        if not manager.socket.closed:
            manager.socket.setsockopt(zmq.LINGER, 0)
            manager.socket.close()

    test_case.addCleanup(close_socket)
    return manager


@skip_without_zmq
class TestStopServer(unittest.TestCase):
    def test_stop_server_sends_shutdown_and_cleans_up(self):
        manager = make_manager()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager.stop_server()
        manager.socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 100)
        manager.socket.send.assert_called_once_with(b"shutdown|null")
        manager.ssh_proc.kill.assert_called_once()
        manager._qdel.assert_called_once()
        manager.socket.setsockopt.assert_any_call(zmq.LINGER, 0)
        manager.socket.close.assert_called_once()
        self.assertTrue(manager.server_stopped)
        self.assertIn("Stopping the remote analysis server", out.getvalue())

    def test_stop_server_is_idempotent(self):
        manager = make_manager()
        with contextlib.redirect_stdout(io.StringIO()):
            manager.stop_server()
            manager.stop_server()
        manager.socket.send.assert_called_once()
        manager._qdel.assert_called_once()
        manager.ssh_proc.kill.assert_called_once()

    def test_stop_server_skips_shutdown_message_if_job_not_running(self):
        manager = make_manager(job_state="Q")
        with contextlib.redirect_stdout(io.StringIO()):
            manager.stop_server()
        manager.socket.send.assert_not_called()
        manager._qdel.assert_called_once()
        manager.ssh_proc.kill.assert_called_once()

    def test_stop_server_deletes_job_when_send_fails(self):
        manager = make_manager()
        manager.socket.send.side_effect = zmq.ZMQError(zmq.EFSM)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager.stop_server()
        manager.ssh_proc.kill.assert_called_once()
        manager._qdel.assert_called_once()
        manager.socket.close.assert_called_once()
        self.assertTrue(manager.server_stopped)
        self.assertIn("Could not send shutdown message", out.getvalue())

    def test_stop_server_deletes_job_even_if_ssh_kill_fails(self):
        manager = make_manager()
        manager.ssh_proc.kill.side_effect = OSError("boom")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                manager.stop_server()
        manager._qdel.assert_called_once()

    def test_stop_server_does_not_kill_already_exited_ssh(self):
        manager = make_manager(ssh_running=False)
        with contextlib.redirect_stdout(io.StringIO()):
            manager.stop_server()
        manager.ssh_proc.kill.assert_not_called()
        manager._qdel.assert_called_once()

    def test_stop_server_at_exit_prints_and_stops(self):
        manager = make_manager()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._stop_server_at_exit()
        self.assertIn("Python exiting; stopping the analysis server", out.getvalue())
        manager._qdel.assert_called_once()
        self.assertTrue(manager.server_stopped)

    def test_stop_server_at_exit_noop_if_already_stopped(self):
        manager = make_manager(server_stopped=True)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._stop_server_at_exit()
        self.assertEqual(out.getvalue(), "")
        manager._qdel.assert_not_called()

    def test_stop_server_at_exit_swallows_errors(self):
        manager = make_manager()
        manager._qdel.side_effect = RuntimeError("qdel failed")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._stop_server_at_exit()
        self.assertIn("Error while stopping server at exit", out.getvalue())


@skip_without_zmq
class TestSshSetup(unittest.TestCase):
    def _run_setup_ssh(self, forward_through_frontend, pbs_o_host):
        manager = make_manager(forward_through_frontend=forward_through_frontend)
        env = {"PBS_O_HOST": pbs_o_host} if pbs_o_host else {}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "mphys.network.zmq_pbs.subprocess.Popen"
        ) as popen:
            manager._setup_ssh()
        popen.assert_called_once()
        args, kwargs = popen.call_args
        return args[0], kwargs

    def test_ssh_command_direct(self):
        argv, kwargs = self._run_setup_ssh(False, "pfe27")
        self.assertEqual(argv[0], "ssh")
        self.assertIn("-N", argv)
        self.assertIn("5081:localhost:5081", argv)
        self.assertNotIn("-J", argv)
        self.assertNotIn("&", argv)
        self.assertEqual(argv[-1], "r101i0n0")
        self.assertIs(kwargs["preexec_fn"], zmq_pbs._terminate_when_parent_dies)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)

    def test_ssh_command_through_frontend(self):
        argv, _ = self._run_setup_ssh(True, "pfe27")
        self.assertIn("-J", argv)
        self.assertEqual(argv[argv.index("-J") + 1], "pfe27")
        self.assertEqual(argv[-1], "r101i0n0")

    def test_ssh_command_frontend_requested_but_unavailable(self):
        argv, _ = self._run_setup_ssh(True, None)
        self.assertNotIn("-J", argv)


@skip_without_zmq
class TestPortSelection(unittest.TestCase):
    def test_port_is_in_use(self):
        manager = make_manager()
        port = free_port()
        self.assertFalse(manager._port_is_in_use(port))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            s.listen(1)
            self.assertTrue(manager._port_is_in_use(port))
        self.assertFalse(manager._port_is_in_use(port))

    def test_initialize_connection_moves_to_free_port(self):
        busy = free_port()
        manager = make_manager(port=busy, acceptable_port_range=[busy, busy + 50])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", busy))
            s.listen(1)
            with mock.patch.object(
                manager, "_initialize_zmq_socket"
            ), contextlib.redirect_stdout(io.StringIO()):
                manager._initialize_connection()
        self.assertNotEqual(manager.port, busy)
        self.assertGreater(manager.port, busy)
        self.assertLessEqual(manager.port, busy + 50)

    def test_initialize_connection_keeps_free_port(self):
        port = free_port()
        manager = make_manager(port=port)
        with mock.patch.object(manager, "_initialize_zmq_socket"):
            manager._initialize_connection()
        self.assertEqual(manager.port, port)

    def test_initialize_connection_raises_if_no_port_available(self):
        busy = free_port()
        manager = make_manager(port=busy, acceptable_port_range=[busy, busy])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", busy))
            s.listen(1)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(zmq_pbs.RemoteComponentError):
                    manager._initialize_connection()


@skip_without_zmq
class TestPbsFailures(unittest.TestCase):
    """PBS commands failing (e.g. unreachable PBS server) must not crash the client."""

    def test_query_job_state_retries_then_succeeds(self):
        manager = make_manager(pbs_retry_attempts=3)
        manager.job.update_job_state.side_effect = [KeyError("Job_Name"), None]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(manager._query_job_state(), "R")
        self.assertEqual(manager.job.update_job_state.call_count, 2)
        self.assertIn("qstat failed", out.getvalue())

    def test_query_job_state_returns_none_after_retries(self):
        manager = make_manager(pbs_retry_attempts=3)
        manager.job.update_job_state.side_effect = KeyError("Job_Name")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(manager._query_job_state())
        self.assertEqual(manager.job.update_job_state.call_count, 3)
        self.assertIn("treating the job state as unknown", out.getvalue())

    def test_job_has_expired_assumes_running_when_pbs_unavailable(self):
        manager = make_manager()
        manager.job.update_job_state.side_effect = KeyError("Job_Name")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(manager.job_has_expired())
        self.assertIn("assuming the server is still running", out.getvalue())
        self.assertEqual(manager.job_expiration_restarts, 0)

    def test_job_has_expired_normal_cases(self):
        manager = make_manager(job_state="R")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(manager.job_has_expired())
            manager.job.state = "F"
            self.assertTrue(manager.job_has_expired())
            manager.job.state = "R"
            manager.server_stopped = True
            self.assertTrue(manager.job_has_expired())

    def test_enough_time_is_remaining_when_pbs_unavailable(self):
        manager = make_manager()
        manager.job.update_job_state.side_effect = KeyError("Job_Name")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(manager.enough_time_is_remaining(1000.0))

    def test_enough_time_is_remaining_normal_cases(self):
        manager = make_manager()
        manager.job.walltime_remaining = 500.0
        self.assertTrue(manager.enough_time_is_remaining(100.0))
        self.assertFalse(manager.enough_time_is_remaining(1000.0))
        manager.job.walltime_remaining = None
        self.assertFalse(manager.enough_time_is_remaining(1.0))

    def test_is_valid_job_id(self):
        valid = zmq_pbs.MPhysZeroMQServerManager._is_valid_job_id
        self.assertTrue(valid("25214052.pbspl1.nas.nasa.gov"))
        self.assertTrue(valid("5517682.pbssrv1\n"))
        self.assertTrue(valid("FakePBS.0"))
        self.assertFalse(valid(""))
        self.assertFalse(valid("qsub: cannot connect to server"))
        self.assertFalse(valid(None))

    def test_submit_job_retries_until_qsub_returns_an_id(self):
        manager = make_manager(pbs_retry_attempts=3)
        manager.pbs.launch.side_effect = ["", OSError("boom"), "123.pbssrv1\n"]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            jobid = manager._submit_job("MPhys5081", ["cmd"])
        self.assertEqual(jobid, "123.pbssrv1")
        self.assertEqual(manager.pbs.launch.call_count, 3)
        self.assertEqual(out.getvalue().count("Job submission failed"), 2)

    def _pbs_manager(self, **attrs):
        manager = make_manager(**attrs)
        manager.pbs = mock.MagicMock(spec=zmq_pbs.PBS)
        manager.pbs.batch_file_extension = "pbs"
        return manager

    def test_qsub_stderr_is_reported(self):
        manager = self._pbs_manager()
        result = subprocess.CompletedProcess(
            ["qsub"], 1, stdout="", stderr="qsub: Unauthorized Request\n"
        )
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", return_value=result
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "Unauthorized Request"
                ):
                    manager._submit_job("MPhys5081", ["cmd"])
        run.assert_called_once()  # a non-transient error is not retried
        self.assertEqual(run.call_args[0][0], ["qsub", "MPhys5081.pbs"])
        manager.pbs.write_job_file.assert_called_once_with(
            "MPhys5081.pbs", "MPhys5081", ["cmd"]
        )

    def test_qsub_transient_error_is_retried(self):
        manager = self._pbs_manager(pbs_retry_attempts=3)
        results = [
            subprocess.CompletedProcess(
                ["qsub"],
                1,
                stdout="",
                stderr="qsub: cannot connect to server pbspl1 (errno=111)",
            ),
            subprocess.CompletedProcess(["qsub"], 0, stdout="123.pbspl1\n", stderr=""),
        ]
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", side_effect=results
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                jobid = manager._submit_job("MPhys5081", ["cmd"])
        self.assertEqual(jobid, "123.pbspl1")
        self.assertEqual(run.call_count, 2)
        self.assertIn("cannot connect to server", out.getvalue())
        self.assertIn("123.pbspl1", out.getvalue())

    def test_qsub_empty_output_is_retried(self):
        manager = self._pbs_manager(pbs_retry_attempts=2)
        result = subprocess.CompletedProcess(["qsub"], 1, stdout="", stderr="")
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", return_value=result
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "after 2 attempts"
                ):
                    manager._submit_job("MPhys5081", ["cmd"])
        self.assertEqual(run.call_count, 2)

    def test_qsub_missing_executable(self):
        manager = self._pbs_manager(pbs_retry_attempts=1)
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run",
            side_effect=FileNotFoundError("qsub"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "FileNotFoundError"
                ):
                    manager._submit_job("MPhys5081", ["cmd"])

    def test_qsub_hang_is_retried(self):
        manager = self._pbs_manager(pbs_retry_attempts=2, pbs_command_timeout=7)
        results = [
            subprocess.TimeoutExpired(["qsub"], 7),
            subprocess.CompletedProcess(["qsub"], 0, stdout="123.pbspl1\n", stderr=""),
        ]
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", side_effect=results
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                jobid = manager._submit_job("MPhys5081", ["cmd"])
        self.assertEqual(jobid, "123.pbspl1")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs["timeout"], 7)
        self.assertIn("did not return within 7 s", out.getvalue())

    def test_qdel_hang_is_retried(self):
        manager = self._real_qdel_manager(qdel_retry_attempts=2, pbs_command_timeout=7)
        results = [subprocess.TimeoutExpired(["qdel"], 7), qdel_result(0)]
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", side_effect=results
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertTrue(manager._qdel())
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs["timeout"], 7)
        self.assertIn("did not return within 7 s", out.getvalue())

    def test_qstat_has_timeout(self):
        with mock.patch("mphys.network.zmq_pbs.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                ["qstat"], 0, stdout=b"Job Id: 123\n    Job_Name = x\n", stderr=b""
            )
            job = zmq_pbs.PBSJobWithTimeout.__new__(zmq_pbs.PBSJobWithTimeout)
            job.id = "123.pbspl1"
            job.qstat_timeout = 7
            lines = job._run_qstat_to_get_full_job_attributes()
        self.assertEqual(run.call_args[0][0], ["qstat", "-xf", "123.pbspl1"])
        self.assertEqual(run.call_args.kwargs["timeout"], 7)
        self.assertIn("    Job_Name = x", lines)

    def test_qstat_hang_is_treated_as_failed_query(self):
        manager = make_manager(pbs_retry_attempts=2)
        manager.job.update_job_state.side_effect = subprocess.TimeoutExpired(
            ["qstat"], 120
        )
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(manager._query_job_state())
        self.assertIn("TimeoutExpired", out.getvalue())

    def test_create_job_handle_uses_timeout_job_class(self):
        manager = make_manager(pbs_command_timeout=7)
        with mock.patch("mphys.network.zmq_pbs.PBSJobWithTimeout") as job_class:
            manager._create_job_handle("123.pbspl1")
        job_class.assert_called_once_with("123.pbspl1")
        self.assertEqual(job_class.qstat_timeout, 7)

    def test_pbs_env_replaces_missing_tmpdir(self):
        with mock.patch.dict(
            os.environ, {"TMPDIR": "/var/tmp/pbs.does_not_exist.pbssrv1"}
        ):
            env = zmq_pbs._pbs_command_env()
        self.assertIn(env.get("TMPDIR"), ("/var/tmp", "/tmp", None))
        self.assertTrue(env.get("TMPDIR") is None or os.path.isdir(env["TMPDIR"]))

    def test_pbs_env_keeps_existing_tmpdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"TMPDIR": tmp}):
                self.assertEqual(zmq_pbs._pbs_command_env()["TMPDIR"], tmp)

    def test_qsub_runs_with_valid_tmpdir(self):
        manager = self._pbs_manager()
        result = subprocess.CompletedProcess(
            ["qsub"], 0, stdout="123.pbssrv1\n", stderr=""
        )
        with mock.patch.dict(
            os.environ, {"TMPDIR": "/var/tmp/pbs.does_not_exist.pbssrv1"}
        ):
            with mock.patch(
                "mphys.network.zmq_pbs.subprocess.run", return_value=result
            ) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    manager._submit_job("MPhys5081", ["cmd"])
        tmpdir = run.call_args.kwargs["env"].get("TMPDIR")
        self.assertTrue(tmpdir is None or os.path.isdir(tmpdir))

    def test_is_transient_qsub_error(self):
        transient = zmq_pbs.MPhysZeroMQServerManager._is_transient_qsub_error
        self.assertTrue(transient("qsub: cannot connect to server pbspl1 (errno=111)"))
        self.assertTrue(transient("Connection refused"))
        self.assertTrue(transient("qsub: Communication failure"))
        self.assertFalse(transient("qsub: Unauthorized Request"))
        self.assertFalse(transient("qsub: Unknown queue"))
        self.assertFalse(transient("qsub: Bad UID for job execution"))

    def test_submit_job_raises_after_retries(self):
        manager = make_manager(pbs_retry_attempts=2)
        manager.pbs.launch.return_value = ""
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(
                zmq_pbs.RemoteComponentError, "Could not submit server job"
            ):
                manager._submit_job("MPhys5081", ["cmd"])
        self.assertEqual(manager.pbs.launch.call_count, 2)

    def test_create_job_handle_retries(self):
        manager = make_manager(pbs_retry_attempts=3)
        fake_job = mock.MagicMock()
        with mock.patch(
            "mphys.network.zmq_pbs.PBSJobWithTimeout",
            side_effect=[KeyError("Job_Name"), fake_job],
        ) as pbsjob:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIs(manager._create_job_handle("123.pbssrv1"), fake_job)
        self.assertEqual(pbsjob.call_count, 2)

    def test_create_job_handle_raises_after_retries(self):
        manager = make_manager(pbs_retry_attempts=2)
        with mock.patch(
            "mphys.network.zmq_pbs.PBSJobWithTimeout", side_effect=KeyError("Job_Name")
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "Could not query PBS"
                ):
                    manager._create_job_handle("123.pbssrv1")

    def test_wait_for_job_to_start_tolerates_qstat_failures(self):
        manager = make_manager(job_state="Q")
        states = iter([KeyError("Job_Name"), "Q", KeyError("Job_Name"), "R"])

        def update():
            s = next(states)
            if isinstance(s, Exception):
                raise s
            manager.job.state = s

        manager.job.update_job_state.side_effect = update
        with mock.patch.multiple(
            manager, _setup_dummy_socket=mock.DEFAULT, _stop_dummy_socket=mock.DEFAULT
        ) as mocks:
            with contextlib.redirect_stdout(io.StringIO()):
                manager._wait_for_job_to_start()
        self.assertEqual(manager.job.state, "R")
        mocks["_setup_dummy_socket"].assert_called_once()
        mocks["_stop_dummy_socket"].assert_called_once()

    def test_wait_for_job_to_start_raises_if_job_fails_in_queue(self):
        manager = make_manager(job_state="Q")
        states = iter(["Q", "F"])

        def update():
            manager.job.state = next(states)

        manager.job.update_job_state.side_effect = update
        manager.job.exit_status = 1
        with mock.patch.multiple(
            manager, _setup_dummy_socket=mock.DEFAULT, _stop_dummy_socket=mock.DEFAULT
        ) as mocks:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "finished before it started"
                ):
                    manager._wait_for_job_to_start()
        mocks["_stop_dummy_socket"].assert_called_once()

    def _real_qdel_manager(self, **attrs):
        manager = make_manager(**attrs)
        del manager._qdel  # use the real method
        return manager

    def test_qdel_success(self):
        manager = self._real_qdel_manager()
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", return_value=qdel_result(0)
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(manager._qdel())
        run.assert_called_once()
        self.assertEqual(run.call_args[0][0], ["qdel", "12345.pbssrv1"])

    def test_qdel_retries_when_pbs_unreachable(self):
        manager = self._real_qdel_manager(qdel_retry_attempts=3)
        results = [
            qdel_result(1, "qdel: cannot connect to server pbspl1 (errno=111)"),
            qdel_result(1, "qdel: cannot connect to server pbspl1 (errno=111)"),
            qdel_result(0),
        ]
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", side_effect=results
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertTrue(manager._qdel())
        self.assertEqual(run.call_count, 3)
        self.assertEqual(out.getvalue().count("qdel failed"), 2)

    def test_qdel_already_finished_job_is_success(self):
        manager = self._real_qdel_manager()
        result = qdel_result(153, "qdel: Unknown Job Id 12345.pbssrv1")
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", return_value=result
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(manager._qdel())
        run.assert_called_once()

    def test_qdel_gives_up_with_warning(self):
        manager = self._real_qdel_manager(qdel_retry_attempts=2)
        result = qdel_result(1, "qdel: cannot connect to server pbspl1 (errno=111)")
        with mock.patch(
            "mphys.network.zmq_pbs.subprocess.run", return_value=result
        ) as run:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertFalse(manager._qdel())
        self.assertEqual(run.call_count, 2)
        self.assertIn("should be deleted manually", out.getvalue())

    def test_stop_server_survives_qdel_failure(self):
        manager = self._real_qdel_manager(qdel_retry_attempts=1)
        result = qdel_result(1, "qdel: cannot connect to server pbspl1 (errno=111)")
        with mock.patch("mphys.network.zmq_pbs.subprocess.run", return_value=result):
            with contextlib.redirect_stdout(io.StringIO()):
                manager.stop_server()
        self.assertTrue(manager.server_stopped)
        manager.ssh_proc.kill.assert_called_once()
        manager.socket.close.assert_called_once()


@skip_without_zmq
class TestStartupHandshake(unittest.TestCase):
    def setUp(self):
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.stop()

    def _handler_ping_only(self, message):
        self.assertEqual(message, b"ping|null")
        return ping_reply()

    def test_ready_immediately(self):
        manager = make_live_manager(self)
        self.server = FakeZmqServer(manager.port, self._handler_ping_only)
        self.server.start()
        self.server.bound.wait(5)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._wait_for_server_to_be_ready()
        self.assertEqual(self.server.received, [b"ping|null"])
        self.assertIn("Server is ready", out.getvalue())
        manager.job.update_job_state.assert_not_called()

    def test_retries_until_server_binds(self):
        # ssh accepts connections before the server has bound its port and the
        # message is lost; the client must keep pinging with fresh sockets
        manager = make_live_manager(self)
        self.server = FakeZmqServer(
            manager.port, self._handler_ping_only, refuse_for=1.0
        )
        self.server.start()
        with mock.patch.object(
            manager, "_reset_zmq_socket", wraps=manager._reset_zmq_socket
        ) as reset:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                manager._wait_for_server_to_be_ready()
        self.assertGreaterEqual(reset.call_count, 2)
        # usually exactly one ping reaches the bound server, but on a slow
        # machine the reply to the first one can miss the ping interval, so the
        # client (correctly) pings again
        self.assertGreaterEqual(len(self.server.received), 1)
        self.assertTrue(all(m == b"ping|null" for m in self.server.received))
        self.assertIn("Server is ready", out.getvalue())

    def test_unexpected_reply_is_ignored(self):
        replies = iter([b'"garbage"', ping_reply()])
        manager = make_live_manager(self)
        self.server = FakeZmqServer(manager.port, lambda m: next(replies))
        self.server.start()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._wait_for_server_to_be_ready()
        self.assertEqual(len(self.server.received), 2)
        self.assertIn("Unexpected reply to ping", out.getvalue())
        self.assertIn("Server is ready", out.getvalue())

    def test_startup_timeout(self):
        manager = make_live_manager(self, startup_timeout=0.5)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(
                zmq_pbs.RemoteComponentError, "did not become ready"
            ):
                manager._wait_for_server_to_be_ready()

    def test_job_dies_during_startup(self):
        manager = make_live_manager(
            self, job_check_interval=0, job_gone_checks_required=2, job_state="F"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(
                zmq_pbs.RemoteComponentError, "ended before the server"
            ):
                manager._wait_for_server_to_be_ready()
        self.assertEqual(manager.job.update_job_state.call_count, 2)

    def test_transient_qstat_failure_does_not_abort_startup(self):
        # qstat occasionally returns nothing; a few such results must not be
        # mistaken for a dead job
        manager = make_live_manager(
            self, job_check_interval=0, job_gone_checks_required=3
        )
        states = iter(["", "", "R", "R", "R", "R", "R", "R"])

        def update():
            manager.job.state = next(states, "R")

        manager.job.update_job_state.side_effect = update
        self.server = FakeZmqServer(
            manager.port, self._handler_ping_only, refuse_for=1.0
        )
        self.server.start()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._wait_for_server_to_be_ready()
        self.assertIn("Server is ready", out.getvalue())

    def test_qstat_failure_during_startup_is_inconclusive(self):
        manager = make_live_manager(
            self, job_check_interval=0, job_gone_checks_required=1, pbs_retry_attempts=1
        )
        manager.job.update_job_state.side_effect = KeyError("Job_Name")
        self.server = FakeZmqServer(
            manager.port, self._handler_ping_only, refuse_for=1.0
        )
        self.server.start()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._wait_for_server_to_be_ready()
        self.assertIn("qstat failed", out.getvalue())
        self.assertIn("Server is ready", out.getvalue())

    def test_start_server_includes_handshake(self):
        manager = make_manager()
        with mock.patch.multiple(
            manager,
            _initialize_connection=mock.DEFAULT,
            _launch_job=mock.DEFAULT,
            _wait_for_server_to_be_ready=mock.DEFAULT,
        ) as mocks:
            manager.server_counter = 0
            manager.start_server()
        mocks["_initialize_connection"].assert_called_once()
        mocks["_launch_job"].assert_called_once()
        mocks["_wait_for_server_to_be_ready"].assert_called_once()
        self.assertFalse(manager.server_stopped)


# env vars that would make a child python try to join the MPI job of an
# mpiexec-launched test process instead of running standalone; MPI
# configuration variables such as OMPI_MCA_* are kept
MPI_ENV_PREFIXES = ("MPI_", "MPT_", "PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")
MPI_CONFIG_PREFIXES = ("OMPI_MCA_", "MPI_ROOT")


def non_mpi_env():
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(MPI_ENV_PREFIXES) or k.startswith(MPI_CONFIG_PREFIXES)
    }


def read_until(stream, predicate, timeout=120):
    """
    Read lines from a subprocess pipe until predicate(line) is true; MPI
    libraries may print warnings (e.g. Open MPI/UCX on CI runners) first.
    Returns (matching_line or None, all lines read).
    """
    lines = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = select.select([stream], [], [], max(0.0, deadline - time.time()))
        if not ready:
            break
        line = stream.readline()
        if line == "":
            break
        lines.append(line)
        if predicate(line):
            return line, lines
    return None, lines


@skip_without_zmq
class TestServerErrorReporting(unittest.TestCase):
    """Run MPhysZeroMQServer for real (in a subprocess) with failing models."""

    MODEL = """
    import numpy as np
    import openmdao.api as om
    from mphys.network.zmq_pbs import MPhysZeroMQServer


    class Fragile(om.ExplicitComponent):
        def setup(self):
            self.add_input("x", 1.0)
            self.add_output("y", 1.0)

        def compute(self, inputs, outputs):
            if inputs["x"][0] > 5:
                raise RuntimeError("solver blew up")
            outputs["y"] = 2 * inputs["x"]


    def get_model():
        model = om.Group()
        model.add_subsystem("comp", Fragile(), promotes=["*"])
        model.add_design_var("x")
        model.add_objective("y")
        return model


    def get_model_with_import_error():
        import module_that_does_not_exist_xyz  # noqa: F401
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.port = free_port()

    def _start_server(self, get_model, report_errors=True):
        script = os.path.join(self.tmp.name, "server.py")
        with open(script, "w") as f:
            f.write(textwrap.dedent(self.MODEL))
            f.write(
                f"\nMPhysZeroMQServer({self.port}, {get_model}, report_errors={report_errors}).run()\n"
            )
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=non_mpi_env(),
            cwd=self.tmp.name,
        )
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://localhost:{self.port}")
        self.addCleanup(sock.close)
        return proc, sock

    def _request(self, sock, command, payload=None, timeout=120):
        sock.send(f"{command}|{json.dumps(payload)}".encode())
        self.assertTrue(sock.poll(timeout * 1000), f"no reply to {command}")
        return sock.recv()

    @staticmethod
    def _inputs(x):
        return {
            "design_vars": {"x": {"val": [x]}},
            "additional_inputs": {},
            "additional_constants": {},
            "additional_outputs": [],
            "component_name": "remote",
        }

    def _error_info(self, reply):
        self.assertTrue(reply.startswith(zmq_pbs.SERVER_ERROR_PREFIX), reply[:80])
        return json.loads(reply)

    def test_startup_error_reported_to_ping(self):
        proc, sock = self._start_server("get_model_with_import_error")
        info = self._error_info(self._request(sock, "ping"))
        out, _ = proc.communicate(timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("module_that_does_not_exist_xyz", info["traceback"])
        self.assertEqual(info["rank"], 0)
        self.assertIn("error reported to client", out)

    def test_runtime_error_reported_as_reply(self):
        proc, sock = self._start_server("get_model")
        self.assertEqual(
            json.loads(self._request(sock, "ping")), zmq_pbs.Server.PING_REPLY
        )
        self._request(sock, "initialize", self._inputs(1.0))
        info = self._error_info(self._request(sock, "evaluate", self._inputs(10.0)))
        proc.communicate(timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        # newer OpenMDAO re-raises as "RuntimeError: '<comp>' ...: Error calling compute(), <msg>"
        self.assertRegex(info["traceback"], r"RuntimeError: .*solver blew up")

    def test_report_errors_disabled(self):
        proc, sock = self._start_server(
            "get_model_with_import_error", report_errors=False
        )
        out, _ = proc.communicate(timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("module_that_does_not_exist_xyz", out)
        self.assertNotIn("error reported to client", out)
        sock.send(b"ping|null")
        self.assertFalse(sock.poll(1000))  # nobody listening


@skip_without_zmq
class TestServerErrorHandlingOnClient(unittest.TestCase):
    ERROR = (
        b'{"status": "error", "host": "r101i0n0", "rank": 0, '
        b'"traceback": "Traceback ...\\nImportError: no module named tacs"}'
    )

    def setUp(self):
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.stop()

    def test_error_reply_to_ping_raises_without_retry(self):
        manager = make_live_manager(self)
        self.server = FakeZmqServer(manager.port, lambda m: self.ERROR)
        self.server.start()
        with mock.patch.object(manager, "stop_server") as stop:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(zmq_pbs.RemoteComponentError) as cm:
                    manager._wait_for_server_to_be_ready()
        stop.assert_called_once()
        self.assertEqual(len(self.server.received), 1)
        self.assertIn("ImportError: no module named tacs", str(cm.exception))
        self.assertIn("on r101i0n0 (rank 0)", str(cm.exception))

    def test_error_reply_to_request_raises_without_resend(self):
        manager = make_live_manager(self)
        self.server = FakeZmqServer(manager.port, lambda m: self.ERROR)
        self.server.start()
        self.server.bound.wait(5)
        manager.send_request(b"evaluate|{}")
        with mock.patch.object(manager, "stop_server") as stop:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "no module named tacs"
                ):
                    manager.receive_reply()
        stop.assert_called_once()
        self.assertEqual(len(self.server.received), 1)

    def test_normal_replies_pass(self):
        manager = make_manager()
        manager._raise_if_server_error(b'{"objective": {}}')
        manager._raise_if_server_error(json.dumps(zmq_pbs.Server.PING_REPLY).encode())

    def _launch_command(self, **attrs):
        manager = make_manager(
            run_server_filename="mphys_server.py",
            additional_server_args="--x 1",
            **attrs,
        )
        manager.server_counter = 1
        manager.pbs.create_mpi_command.side_effect = lambda cmd, output_root_name: cmd
        with mock.patch.multiple(
            manager,
            _submit_job=mock.DEFAULT,
            _create_job_handle=mock.DEFAULT,
            _wait_for_job_to_start=mock.DEFAULT,
            _setup_ssh=mock.DEFAULT,
        ) as mocks:
            with contextlib.redirect_stdout(io.StringIO()):
                manager._launch_job()
        return mocks["_submit_job"].call_args[0][1][0], manager

    def test_launch_command(self):
        command, manager = self._launch_command()
        self.assertEqual(command.split()[:3], ["python", "mphys_server.py", "--port"])
        self.assertTrue(command.rstrip().endswith("--x 1"))
        self.assertTrue(manager._server_output_file.endswith("mphys_test_server1.out"))

    def test_startup_failure_includes_server_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_file = os.path.join(tmp, "mphys_test_server1.out")
            with open(out_file, "w") as f:
                f.write("line 1\nSegmentation fault (core dumped)\n")
            manager = make_live_manager(
                self, job_check_interval=0, job_gone_checks_required=1, job_state="F"
            )
            manager._server_output_file = out_file
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "(?s)ended before.*Segmentation fault"
                ):
                    manager._wait_for_server_to_be_ready()


@skip_without_zmq
class TestReceiveReplyRetry(unittest.TestCase):
    REQUEST = b'evaluate|{"design_vars": {"x": {"val": [1.0]}}}'
    REPLY = b'{"objective": {"f": {"val": [1.0]}}}'

    def setUp(self):
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.stop()

    def _start_server(self, manager, handler):
        self.server = FakeZmqServer(manager.port, handler)
        self.server.start()
        self.server.bound.wait(5)

    def test_reply_received_normally(self):
        manager = make_live_manager(self)
        self._start_server(manager, lambda m: self.REPLY)
        manager.send_request(self.REQUEST)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            reply = manager.receive_reply()
        self.assertEqual(reply, self.REPLY)
        self.assertEqual(self.server.received, [self.REQUEST])
        self.assertEqual(out.getvalue(), "")
        manager.job.update_job_state.assert_not_called()

    def test_lost_reply_is_recovered_by_resending(self):
        drop_first = iter([None, self.REPLY])
        manager = make_live_manager(self)
        self._start_server(manager, lambda m: next(drop_first))
        manager.send_request(self.REQUEST)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            reply = manager.receive_reply()
        self.assertEqual(reply, self.REPLY)
        self.assertEqual(self.server.received, [self.REQUEST, self.REQUEST])
        self.assertIn("re-sending request in case it was lost", out.getvalue())
        manager.job.update_job_state.assert_called()

    def test_slow_server_reply_still_accepted(self):
        # a reply that arrives after a resend (server was just slow) is fine:
        # both server replies are for the same request
        def slow_then_fast(message):
            if len(self.server.received) == 1:
                time.sleep(0.5)  # longer than receive_timeout
            return self.REPLY

        manager = make_live_manager(self)
        self._start_server(manager, slow_then_fast)
        manager.send_request(self.REQUEST)
        with contextlib.redirect_stdout(io.StringIO()):
            reply = manager.receive_reply()
        self.assertEqual(reply, self.REPLY)

    def test_dead_job_triggers_restart_and_resend(self):
        manager = make_live_manager(self, job_gone_checks_required=1)
        replies = iter([None, self.REPLY])
        self._start_server(manager, lambda m: next(replies))

        def fake_stop():
            manager.server_stopped = True
            manager.socket.setsockopt(zmq.LINGER, 0)
            manager.socket.close()

        def fake_start():
            manager.job.state = "R"
            manager._initialize_zmq_socket()
            manager.server_stopped = False

        manager.job.state = "F"
        manager.send_request(self.REQUEST)
        with mock.patch.object(manager, "stop_server", side_effect=fake_stop) as stop:
            with mock.patch.object(
                manager, "start_server", side_effect=fake_start
            ) as start:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    reply = manager.receive_reply()
        self.assertEqual(reply, self.REPLY)
        stop.assert_called_once()
        start.assert_called_once()
        self.assertIn("Server job ended while waiting", out.getvalue())
        self.assertEqual(self.server.received, [self.REQUEST, self.REQUEST])

    def test_dead_job_requires_consecutive_checks(self):
        manager = make_live_manager(self, job_gone_checks_required=3)
        states = iter(["", "", "R"])

        def update():
            manager.job.state = next(states, "R")

        manager.job.update_job_state.side_effect = update
        replies = iter([None, None, None, self.REPLY])
        self._start_server(manager, lambda m: next(replies))
        manager.send_request(self.REQUEST)
        with mock.patch.object(manager, "stop_server") as stop:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                reply = manager.receive_reply()
        self.assertEqual(reply, self.REPLY)
        stop.assert_not_called()
        self.assertIn("(1/3 checks)", out.getvalue())
        self.assertIn("(2/3 checks)", out.getvalue())

    def test_pbs_unavailable_keeps_waiting_and_resending(self):
        manager = make_live_manager(
            self, job_gone_checks_required=1, pbs_retry_attempts=1
        )
        manager.job.update_job_state.side_effect = KeyError("Job_Name")
        replies = iter([None, None, self.REPLY])
        self._start_server(manager, lambda m: next(replies))
        manager.send_request(self.REQUEST)
        with mock.patch.object(manager, "stop_server") as stop:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                reply = manager.receive_reply()
        self.assertEqual(reply, self.REPLY)
        stop.assert_not_called()
        self.assertEqual(len(self.server.received), 3)
        self.assertIn("qstat failed", out.getvalue())
        self.assertIn("re-sending request", out.getvalue())

    def test_max_restarts_enforced(self):
        manager = make_live_manager(
            self,
            job_gone_checks_required=1,
            job_expiration_max_restarts=1,
            job_expiration_restarts=1,
        )
        self._start_server(manager, lambda m: None)
        manager.job.state = "F"
        manager.send_request(self.REQUEST)
        with mock.patch.object(manager, "stop_server") as stop:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    zmq_pbs.RemoteComponentError, "maximum number"
                ):
                    manager.receive_reply()
        stop.assert_called_once()


@skip_without_zmq
class TestSignalHandling(unittest.TestCase):
    def setUp(self):
        self.saved = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)
        }

    def tearDown(self):
        for sig, handler in self.saved.items():
            signal.signal(sig, handler)

    def test_default_handlers_are_replaced(self):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        zmq_pbs._install_shutdown_signal_handlers()
        self.assertIs(signal.getsignal(signal.SIGTERM), zmq_pbs._exit_on_signal)
        self.assertIs(signal.getsignal(signal.SIGHUP), zmq_pbs._exit_on_signal)

    def test_user_handlers_are_preserved(self):
        def user_handler(signum, frame):
            pass

        signal.signal(signal.SIGTERM, user_handler)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        zmq_pbs._install_shutdown_signal_handlers()
        self.assertIs(signal.getsignal(signal.SIGTERM), user_handler)
        self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)

    def test_exit_on_signal_raises_system_exit(self):
        # handler writes to sys.stdout's file descriptor directly, so give it a
        # real file object backed by a pipe (test runners may replace sys.stdout)
        read_fd, write_fd = os.pipe()
        with os.fdopen(write_fd, "w") as fake_stdout:
            with mock.patch("sys.stdout", fake_stdout):
                with self.assertRaises(SystemExit) as cm:
                    zmq_pbs._exit_on_signal(signal.SIGTERM, None)
        out = os.read(read_fd, 4096).decode()
        os.close(read_fd)
        self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)
        self.assertIn("Received signal SIGTERM", out)

    def test_exit_on_signal_tolerates_missing_stdout_fd(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                zmq_pbs._exit_on_signal(signal.SIGHUP, None)
        self.assertEqual(cm.exception.code, 128 + signal.SIGHUP)


@skip_without_zmq
class TestProcessLifetime(unittest.TestCase):
    """
    Spawn a real client-like python process (no PBS/zmq traffic) and check
    that shutdown hooks fire and that a child with the death signal set dies
    with its parent.
    """

    CLIENT_SCRIPT = textwrap.dedent(
        """
        import atexit, subprocess, sys, time
        from unittest import mock
        import mphys.network.zmq_pbs as zmq_pbs

        manager = zmq_pbs.MPhysZeroMQServerManager.__new__(zmq_pbs.MPhysZeroMQServerManager)
        manager.component_name = "test"
        manager.shutdown_send_timeout_ms = 100
        manager.job = mock.MagicMock(); manager.job.state = "Q"
        manager.socket = mock.MagicMock()
        manager.ssh_proc = subprocess.Popen(
            ["sleep", "300"], preexec_fn=zmq_pbs._terminate_when_parent_dies
        )
        manager.server_stopped = False
        zmq_pbs._install_shutdown_signal_handlers()
        atexit.register(manager._stop_server_at_exit)
        print(manager.ssh_proc.pid, flush=True)
        time.sleep(300)
        """
    )

    def _spawn_client(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", self.CLIENT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=non_mpi_env(),
        )
        pid_line, lines = read_until(proc.stdout, lambda line: line.strip().isdigit())
        if pid_line is None:
            proc.kill()
            out, _ = proc.communicate()
            self.fail(
                "client subprocess did not report its child pid; output:\n"
                + "".join(lines)
                + out
            )
        return proc, int(pid_line)

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _wait_for_death(self, pid, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._pid_alive(pid):
                return True
            time.sleep(0.05)
        return False

    def test_sigterm_runs_stop_server(self):
        proc, child_pid = self._spawn_client()
        try:
            proc.send_signal(signal.SIGTERM)
            out, _ = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM, out)
        self.assertIn("Received signal SIGTERM", out)
        self.assertIn("Python exiting; stopping the analysis server", out)
        self.assertIn("Stopping the remote analysis server", out)
        self.assertTrue(self._wait_for_death(child_pid))

    def test_sigterm_while_writing_to_stdout(self):
        # the signal handler must not use buffered stdout: if the signal lands
        # while the main thread is inside a print, that raises RuntimeError
        # (reentrant call) and replaces SystemExit with exit code 1
        script = textwrap.dedent(
            """
            import sys
            import mphys.network.zmq_pbs as zmq_pbs
            zmq_pbs._install_shutdown_signal_handlers()
            sys.stderr.write("READY\\n"); sys.stderr.flush()
            while True:
                print("x" * 65536, flush=True)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=non_mpi_env(),
        )
        try:
            ready_line, lines = read_until(
                proc.stderr, lambda line: line.strip() == "READY"
            )
            self.assertIsNotNone(ready_line, "client did not start:\n" + "".join(lines))
            time.sleep(0.5)  # let the child block inside a flush on the full pipe
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 128 + signal.SIGTERM, err)
        self.assertNotIn("reentrant", err)
        self.assertIn("Received signal SIGTERM", out)

    def test_sigint_runs_stop_server(self):
        proc, child_pid = self._spawn_client()
        try:
            proc.send_signal(signal.SIGINT)
            out, _ = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertIn("KeyboardInterrupt", out)
        self.assertIn("Python exiting; stopping the analysis server", out)
        self.assertTrue(self._wait_for_death(child_pid))

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux-only"
    )
    def test_ssh_child_dies_when_client_is_sigkilled(self):
        proc, child_pid = self._spawn_client()
        self.assertTrue(self._pid_alive(child_pid))
        proc.kill()
        proc.communicate(timeout=20)
        self.assertTrue(self._wait_for_death(child_pid))


if __name__ == "__main__":
    unittest.main()
