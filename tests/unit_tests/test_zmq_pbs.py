import contextlib
import io
import os
import select
import signal
import socket
import subprocess
import sys
import textwrap
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
    manager = zmq_pbs.MPhysZeroMQServerManager.__new__(
        zmq_pbs.MPhysZeroMQServerManager
    )
    manager.component_name = "test"
    manager.port = 5081
    manager.acceptable_port_range = [5081, 5090]
    manager.forward_through_frontend = False
    manager.shutdown_send_timeout_ms = 100
    manager.server_stopped = False
    manager.job = mock.MagicMock()
    manager.job.state = job_state
    manager.job.hostname = "r101i0n0"
    manager.socket = mock.MagicMock()
    manager.ssh_proc = mock.MagicMock()
    manager.ssh_proc.poll.return_value = None if ssh_running else 0
    for key, val in attrs.items():
        setattr(manager, key, val)
    return manager


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@skip_without_zmq
class TestStopServer(unittest.TestCase):
    N_PROCS = 1

    def test_stop_server_sends_shutdown_and_cleans_up(self):
        manager = make_manager()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager.stop_server()
        manager.socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 100)
        manager.socket.send.assert_called_once_with(b"shutdown|null")
        manager.ssh_proc.kill.assert_called_once()
        manager.job.qdel.assert_called_once()
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
        manager.job.qdel.assert_called_once()
        manager.ssh_proc.kill.assert_called_once()

    def test_stop_server_skips_shutdown_message_if_job_not_running(self):
        manager = make_manager(job_state="Q")
        with contextlib.redirect_stdout(io.StringIO()):
            manager.stop_server()
        manager.socket.send.assert_not_called()
        manager.job.qdel.assert_called_once()
        manager.ssh_proc.kill.assert_called_once()

    def test_stop_server_deletes_job_when_send_fails(self):
        manager = make_manager()
        manager.socket.send.side_effect = zmq.ZMQError(zmq.EFSM)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager.stop_server()
        manager.ssh_proc.kill.assert_called_once()
        manager.job.qdel.assert_called_once()
        manager.socket.close.assert_called_once()
        self.assertTrue(manager.server_stopped)
        self.assertIn("Could not send shutdown message", out.getvalue())

    def test_stop_server_deletes_job_even_if_ssh_kill_fails(self):
        manager = make_manager()
        manager.ssh_proc.kill.side_effect = OSError("boom")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                manager.stop_server()
        manager.job.qdel.assert_called_once()

    def test_stop_server_does_not_kill_already_exited_ssh(self):
        manager = make_manager(ssh_running=False)
        with contextlib.redirect_stdout(io.StringIO()):
            manager.stop_server()
        manager.ssh_proc.kill.assert_not_called()
        manager.job.qdel.assert_called_once()

    def test_stop_server_at_exit_prints_and_stops(self):
        manager = make_manager()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._stop_server_at_exit()
        self.assertIn("Python exiting; stopping the analysis server", out.getvalue())
        manager.job.qdel.assert_called_once()
        self.assertTrue(manager.server_stopped)

    def test_stop_server_at_exit_noop_if_already_stopped(self):
        manager = make_manager(server_stopped=True)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._stop_server_at_exit()
        self.assertEqual(out.getvalue(), "")
        manager.job.qdel.assert_not_called()

    def test_stop_server_at_exit_swallows_errors(self):
        manager = make_manager()
        manager.job.qdel.side_effect = RuntimeError("qdel failed")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            manager._stop_server_at_exit()
        self.assertIn("Error while stopping server at exit", out.getvalue())


@skip_without_zmq
class TestSshSetup(unittest.TestCase):
    N_PROCS = 1

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
    N_PROCS = 1

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
            with mock.patch.object(manager, "_initialize_zmq_socket"), contextlib.redirect_stdout(
                io.StringIO()
            ):
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
                with self.assertRaises(RuntimeError):
                    manager._initialize_connection()


@skip_without_zmq
class TestSignalHandling(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        self.saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}

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

    N_PROCS = 1

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

    # env vars that would make the child python try to join the MPI job of
    # the (mpiexec-launched) test process instead of running standalone
    MPI_ENV_PREFIXES = ("MPI_", "MPT_", "PMI_", "PMIX_", "OMPI_", "MPICH_", "HYDRA_")

    def _spawn_client(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(self.MPI_ENV_PREFIXES)
        }
        proc = subprocess.Popen(
            [sys.executable, "-c", self.CLIENT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        ready, _, _ = select.select([proc.stdout], [], [], 120)
        if not ready:
            proc.kill()
            out, _ = proc.communicate()
            self.fail(f"client subprocess did not start in time; output:\n{out}")
        first_line = proc.stdout.readline()
        try:
            child_pid = int(first_line)
        except ValueError:
            proc.kill()
            out, _ = proc.communicate()
            self.fail(f"unexpected client output: {first_line!r}\n{out}")
        return proc, child_pid

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
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(self.MPI_ENV_PREFIXES)
        }
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            ready, _, _ = select.select([proc.stderr], [], [], 120)
            self.assertTrue(ready, "client subprocess did not start in time")
            self.assertEqual(proc.stderr.readline().strip(), "READY")
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

    @unittest.skipUnless(sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux-only")
    def test_ssh_child_dies_when_client_is_sigkilled(self):
        proc, child_pid = self._spawn_client()
        self.assertTrue(self._pid_alive(child_pid))
        proc.kill()
        proc.communicate(timeout=20)
        self.assertTrue(self._wait_for_death(child_pid))


if __name__ == "__main__":
    unittest.main()
