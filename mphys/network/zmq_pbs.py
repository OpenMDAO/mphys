import argparse
import json
import socket
import subprocess
import time
import zmq

from pbs4py import PBS
from pbs4py.job import PBSJob
from mphys.network import RemoteComp, Server, ServerManager

class RemoteZeroMQComp(RemoteComp):
    """
    A derived RemoteComp class that uses pbs4py for HPC job management
    and ZeroMQ for network communication.
    """
    def initialize(self):
        self.options.declare('pbs', "pbs4py Launcher object")
        self.options.declare('port', default=5081, desc="port number for server/client communication")
        self.options.declare('acceptable_port_range', default=[5081,6000], desc="port range to look through if 'port' is currently busy")
        self.options.declare('additional_server_args', default="", desc="Optional arguments to give server, in addition to --port <port number>")
        super().initialize()
        self.server_manager = None # for avoiding reinitialization due to multiple setup calls

    def _send_inputs_to_server(self, remote_input_dict, command: str):
        if self._doing_derivative_evaluation(command):
            print(f'CLIENT (subsystem {self.name}): Requesting derivative call from server', flush=True)
        else:
            print(f'CLIENT (subsystem {self.name}): Requesting function call from server', flush=True)
        input_str = f"{command}|{str(json.dumps(remote_input_dict))}"
        self.server_manager.socket.send(input_str.encode())

    def _receive_outputs_from_server(self):
        return json.loads(self.server_manager.socket.recv().decode())

    def _setup_server_manager(self):
        if self.server_manager is None:
            self.server_manager = MPhysZeroMQServerManager(pbs=self.options['pbs'],
                                                           run_server_filename=self.options['run_server_filename'],
                                                           component_name=self.name,
                                                           port=self.options['port'],
                                                           acceptable_port_range=self.options['acceptable_port_range'],
                                                           additional_server_args=self.options['additional_server_args'])

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
    additional_server_args : str
        Optional arguments to give server, in addition to --port <port number>
    """
    def __init__(self,
                 pbs: PBS,
                 run_server_filename: str,
                 component_name: str,
                 port=5081,
                 acceptable_port_range=[5081,6000],
                 additional_server_args=''
                 ):
        self.pbs = pbs
        self.run_server_filename = run_server_filename
        self.component_name = component_name
        self.port = port
        self.acceptable_port_range = acceptable_port_range
        self.additional_server_args = additional_server_args
        self.queue_time_delay = 5 # seconds to wait before rechecking if a job has started
        self.server_counter = 0 # for saving output of each server to different files
        self.start_server()

    def start_server(self):
        self._initialize_connection()
        self.server_counter += 1
        self._launch_job()

    def stop_server(self):
        print(f'CLIENT (subsystem {self.component_name}): Stopping the remote analysis server', flush=True)
        self.socket.send('shutdown|null'.encode())
        self._shutdown_server()
        self.socket.close()

    def enough_time_is_remaining(self, estimated_model_time):
        self.job.update_job_state()
        if self.job.walltime_remaining is None:
            return False
        else:
            return estimated_model_time < self.job.walltime_remaining

    def _port_is_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port))==0

    def _initialize_connection(self):
        if self._port_is_in_use(self.port):
            print(f'CLIENT (subsystem {self.component_name}): Port {self.port} is already in use... finding first available port in the range {self.acceptable_port_range}', flush=True)

            for port in range(self.acceptable_port_range[0],self.acceptable_port_range[1]+1):
                if not self._port_is_in_use(port):
                    self.port = port
                    break
            else:
                raise RuntimeError(f'CLIENT (subsystem {self.component_name}): Could not find open port')

        self._initialize_zmq_socket()

    def _initialize_zmq_socket(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")

    def _launch_job(self):
        print(f'CLIENT (subsystem {self.component_name}): Launching new server', flush=True)
        python_command = (f"python {self.run_server_filename} --port {self.port} {self.additional_server_args}")
        python_mpi_command = self.pbs.create_mpi_command(python_command, output_root_name=f'mphys_{self.component_name}_server{self.server_counter}')
        jobid = self.pbs.launch(f'MPhys{self.port}', [python_mpi_command], blocking=False)
        self.job = PBSJob(jobid)
        self._wait_for_job_to_start()
        self._setup_ssh()

    def _wait_for_job_to_start(self):
        print(f'CLIENT (subsystem {self.component_name}): Waiting for job to start', flush=True)
        job_submission_time = time.time()
        self._setup_placeholder_ssh()
        while self.job.state!='R':
            time.sleep(self.queue_time_delay)
            self.job.update_job_state()
        self._stop_placeholder_ssh()
        self.job_start_time = time.time()
        print(f'CLIENT (subsystem {self.component_name}): Job started (queue wait time: {(time.time()-job_submission_time)/3600} hours)', flush=True)

    def _setup_ssh(self):
        ssh_command = f'ssh -4 -o ServerAliveCountMax=40 -o ServerAliveInterval=15 -N -L {self.port}:localhost:{self.port} {self.job.hostname} &'
        self.ssh_proc = subprocess.Popen(ssh_command.split(),
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)

    def _shutdown_server(self):
        self.ssh_proc.kill()
        time.sleep(0.1) # prevent full shutdown before job deletion?
        self.job.qdel()

    def _setup_placeholder_ssh(self):
        print(f'CLIENT (subsystem {self.component_name}): Starting placeholder process to hold port {self.port} while in queue', flush=True)
        ssh_command = f'ssh -4 -o ServerAliveCountMax=40 -o ServerAliveInterval=15 -N -L {self.port}:localhost:{self.port} {socket.gethostname()} &'
        self.ssh_proc = subprocess.Popen(ssh_command.split(),
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)

    def _stop_placeholder_ssh(self):
        self.ssh_proc.kill()

class MPhysZeroMQServer(Server):
    """
    A derived Server class that uses ZeroMQ for network communication.
    """
    def __init__(self, port, get_om_group_function_pointer,
                 ignore_setup_warnings = False,
                 ignore_runtime_warnings = False,
                 rerun_initial_design = False):

        super().__init__(get_om_group_function_pointer, ignore_setup_warnings,
                         ignore_runtime_warnings, rerun_initial_design)
        self._setup_zeromq_socket(port)

    def _setup_zeromq_socket(self, port):
        if self.rank==0:
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{port}")

    def _parse_incoming_message(self):
        inputs = None
        if self.rank==0:
            inputs = self.socket.recv().decode()
        inputs = self.prob.model.comm.bcast(inputs)

        command, input_dict = inputs.split("|")
        if command!='shutdown':
            input_dict = json.loads(input_dict)
        return command, input_dict

    def _send_outputs_to_client(self, output_dict: dict):
        if self.rank==0:
            self.socket.send(str(json.dumps(output_dict)).encode())

def get_default_zmq_pbs_argparser():
    parser = argparse.ArgumentParser('Python script for launching mphys analysis server',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, help='tcp port number for zeromq socket')
    return parser
