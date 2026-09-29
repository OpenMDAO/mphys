"""
In-process stand-ins for the remote component machinery so that RemoteComp and
Server can be exercised without PBS, ssh, or ZeroMQ. Messages are still
round-tripped through JSON strings to mimic the real transport.
"""

import json

import numpy as np
import openmdao.api as om
from mpi4py import MPI

from mphys.network import RemoteComp, Server, ServerManager


def get_paraboloid_group():
    """
    Server-side model. Promoted names are used as-is by the client.
      f = x0^2 + x1^2 + y^2 + p          (objective)
      g = x0 + x1 + c                    (inequality constraint)
      h = y - x0                         (equality constraint)
      z = 3*x + y                        (additional output, vector)
    with design vars x (2-vector, ref/ref0) and y (scaler/adder),
    additional input p and additional constant c.
    """
    model = om.Group()
    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("x", val=np.array([1.0, 2.0]), units="m")
    ivc.add_output("y", val=3.0)
    ivc.add_output("p", val=0.5, units="kg")
    ivc.add_output("c", val=10.0)
    model.add_subsystem(
        "calc",
        om.ExecComp(
            [
                "f = x[0]**2 + x[1]**2 + y**2 + p",
                "g = x[0] + x[1] + c",
                "h = y - x[0]",
                "z = 3*x + y",
            ],
            x={"shape": 2, "units": "m"},
            p={"units": "kg"},
            z={"shape": 2},
        ),
        promotes=["*"],
    )
    model.add_design_var(
        "x", lower=-10.0, upper=10.0, ref=2.0, ref0=-1.0, units="m"
    )
    model.add_design_var("y", lower=-5.0, upper=5.0, scaler=2.0, adder=1.0)
    model.add_objective("f", ref=100.0)
    model.add_constraint("g", upper=20.0)
    model.add_constraint("h", equals=1.0, ref=3.0)
    return model


def get_nested_group():
    """
    Server-side model whose variable names contain dots, to test the
    var_naming_dot_replacement option.
    """
    model = om.Group()
    sub = model.add_subsystem("sub", om.Group())
    ivc = sub.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("x", val=2.0)
    sub.add_subsystem("calc", om.ExecComp("f = x**2"), promotes=["*"])
    model.add_design_var("sub.x", lower=-1.0, upper=1.0)
    model.add_objective("sub.f")
    return model


class InProcessServer(Server):
    """
    A Server that processes one JSON message per call to handle(); each call
    runs Server.run() until the injected shutdown message is reached.
    """

    def __init__(self, *args, **kwargs):
        self._pending = []
        self._response = None
        self.messages_received = []
        super().__init__(*args, **kwargs)

    def _load_the_model(self):
        self.prob = om.Problem(comm=MPI.COMM_SELF)
        self.prob.model = self.get_om_group_function_pointer()
        self.prob.setup(mode="rev")
        self.comm = self.prob.model.comm

    def handle(self, message: str) -> str:
        self._pending = [message, "shutdown|null"]
        self.run()
        return self._response

    def _parse_incoming_message(self):
        message = self._pending.pop(0)
        command, input_str = message.split("|", 1)
        if command == "shutdown":
            return command, None
        self.messages_received.append(command)
        return command, json.loads(input_str)

    def _send_outputs_to_client(self, output_dict):
        self._response = json.dumps(output_dict)


class UnreachableServer:
    """A server that fails the test if the client ever contacts it."""

    def handle(self, message: str):
        raise AssertionError(f"Client contacted the server unexpectedly: {message[:40]}")


class RecordingServerManager(ServerManager):
    """
    A ServerManager that records start/stop calls and lets tests control
    whether the job appears expired or short on walltime.
    """

    def __init__(self, time_remaining=True):
        self.time_remaining = time_remaining
        self.start_calls = 0
        self.stop_calls = 0
        self.stopped = False

    def start_server(self):
        self.start_calls += 1
        self.stopped = False

    def stop_server(self):
        self.stop_calls += 1
        self.stopped = True

    def enough_time_is_remaining(self, estimated_model_time):
        return self.time_remaining

    def job_has_expired(self):
        return self.stopped


class InProcessRemoteComp(RemoteComp):
    """
    A RemoteComp that talks to an in-process server object directly.

    Options
    -------
    server_factory : callable returning an object with a handle(str) -> str method
    server_manager : ServerManager instance to use (default: RecordingServerManager)
    """

    def initialize(self):
        self.options.declare("server_factory")
        self.options.declare("server_manager", default=None)
        super().initialize()
        self.server_manager = None
        self.server = None
        self._response = None

    def _setup_server_manager(self):
        self.server_manager = self.options["server_manager"]
        if self.server_manager is None:
            self.server_manager = RecordingServerManager()
        self.server = self.options["server_factory"]()

    def _send_inputs_to_server(self, remote_input_dict, command: str):
        message = f"{command}|{json.dumps(remote_input_dict)}"
        self._response = self.server.handle(message)

    def _receive_outputs_from_server(self):
        return json.loads(self._response)
