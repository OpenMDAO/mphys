import zmq
import subprocess, time, socket

from pbs4py import PBS
from pbs4py.job import PBSJob

queue_time_delay = 5 # seconds to wait before rechecking if a job has started

class MPhysZeroMQServerManager:
    def __init__(self,
                 pbs: PBS,
                 run_server_filename: str,
                 port=5080,
                 acceptable_port_range=[5080,6000]
                 ):
        self.pbs = pbs
        self.run_server_filename = run_server_filename
        self.port = port
        self.acceptable_port_range = acceptable_port_range
        self.start_server()

    def start_server(self):
        self._initialize_connection()
        self._launch_job()

    def stop_server(self):
        print('CLIENT: Stopping the MPhys analysis server', flush=True)

        # send shutdown signal to server
        self.socket.send('shutdown|null'.encode())

        # close ssh connection
        self._shutdown_server()

        # close zeromq socket
        self.socket.close()

    def enough_time_is_remaining(self, estimated_model_time):
        self.job.update_job_state()
        return estimated_model_time < self.job.walltime_remaining

    def _port_is_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port))==0

    def _initialize_connection(self):
        # first check if port is in use
        if self._port_is_in_use(self.port):
            print(f'CLIENT: Port {self.port} is already in use... finding first available port in the range {self.acceptable_port_range}', flush=True)

            for port in range(self.acceptable_port_range[0],self.acceptable_port_range[1]+1):
                if not self._port_is_in_use(port):
                    self.port = port
                    break
            else:
                raise RuntimeError('CLIENT: Could not find open port')

        # initialize zmq socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")

    def _launch_job(self):
        print('CLIENT: Launching new server', flush=True)
        python_command = (f"python {self.run_server_filename} --port {self.port}")
        python_mpi_command = self.pbs.create_mpi_command(python_command, output_root_name='mphys_server')
        jobid = self.pbs.launch('MPhys_server', [python_mpi_command], blocking=False)
        self.job = PBSJob(jobid)
        self._wait_for_job_to_start()
        self._setup_ssh()

    def _wait_for_job_to_start(self):
        print('CLIENT: Waiting for job to start', flush=True)
        self._setup_placeholder_ssh()
        while self.job.state=='Q':
            time.sleep(queue_time_delay)
            self.job.update_job_state()
        self._stop_placeholder_ssh()
        self.job_start_time = time.time()
        print('CLIENT: Job started', flush=True)

    def _setup_ssh(self):
        ssh_command = f'ssh -4 -o ServerAliveCountMax=40 -o ServerAliveInterval=15 -N -L {self.port}:localhost:{self.port} {self.job.hostname} &'
        self.ssh_proc = subprocess.Popen(ssh_command.split(),
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)

    def _shutdown_server(self):
        self.ssh_proc.kill()
        self.job.qdel()

    def _setup_placeholder_ssh(self):
        print(f'CLIENT: Starting placeholder process to hold port {self.port} while in queue', flush=True)
        ssh_command = f'ssh -4 -o ServerAliveCountMax=40 -o ServerAliveInterval=15 -N -L {self.port}:localhost:{self.port} {socket.gethostname()} &'
        self.ssh_proc = subprocess.Popen(ssh_command.split(),
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)

    def _stop_placeholder_ssh(self):
        self.ssh_proc.kill()
