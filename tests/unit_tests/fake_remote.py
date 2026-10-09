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
      k = 2*x0 - y                       (linear inequality constraint)
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
                "k = 2*x[0] - y",
                "z = 3*x + y",
            ],
            x={"shape": 2, "units": "m"},
            p={"units": "kg"},
            z={"shape": 2},
        ),
        promotes=["*"],
    )
    model.add_design_var("x", lower=-10.0, upper=10.0, ref=2.0, ref0=-1.0, units="m")
    model.add_design_var("y", lower=-5.0, upper=5.0, scaler=2.0, adder=1.0)
    model.add_objective("f", ref=100.0)
    model.add_constraint("g", upper=20.0)
    model.add_constraint("h", equals=1.0, ref=3.0)
    model.add_constraint("k", lower=-20.0, upper=20.0, linear=True)
    return model


def get_parallel_scenarios_group():
    """
    Server-side model shaped like the supersonic panel as_opt_parallel.py
    example: two scenarios evaluated in a ParallelGroup (one per rank when run
    on 2 procs), with constraints from both scenarios sharing parallel
    derivative colors, an objective outside the parallel group, an
    additional input feeding both scenarios, and an additional output from
    the second scenario.
    """
    model = om.Group()
    ivc = model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("x", val=np.array([1.0, 2.0]))
    ivc.add_output("y", val=0.5)
    ivc.add_output("p", val=1.5)
    model.add_subsystem(
        "mass", om.ExecComp("m = x[0]**2 + x[1]**2", x={"shape": 2}), promotes=["*"]
    )
    par = model.add_subsystem("par", om.ParallelGroup(), promotes_inputs=["*"])
    par.add_subsystem(
        "scen0",
        om.ExecComp(
            ["c = x[0]*y + p", "s = x[0]**2 + x[1]", "extra = 3*x[1]*y"],
            x={"shape": 2},
        ),
        promotes_inputs=["*"],
    )
    par.add_subsystem(
        "scen1",
        om.ExecComp(
            ["c = x[1]*y**2 - p", "s = x[0]*x[1]", "extra = x[0] + 2*y*p"],
            x={"shape": 2},
        ),
        promotes_inputs=["*"],
    )
    model.add_design_var("x", lower=-5.0, upper=5.0)
    model.add_design_var("y", lower=-5.0, upper=5.0, ref=2.0)
    model.add_objective("m", ref=10.0)
    for i in range(2):
        model.add_constraint(f"par.scen{i}.c", lower=0.1, parallel_deriv_color="lift")
        model.add_constraint(f"par.scen{i}.s", upper=4.0, parallel_deriv_color="stress")
    return model


PARALLEL_SCENARIO_RESPONSES = [
    "m",
    "par.scen0.c",
    "par.scen1.c",
    "par.scen0.s",
    "par.scen1.s",
]
PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS = ["par.scen1.extra"]
PARALLEL_SCENARIO_ADDITIONAL_INPUTS = ["p"]


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

    By default the server model runs on COMM_SELF. With comm=MPI.COMM_WORLD
    it runs on all ranks: handle() must then be called collectively, the
    message given on rank 0 is broadcast to the other ranks (as in
    MPhysZeroMQServer), and the reply is only meaningful on rank 0.
    """

    def __init__(self, *args, comm=None, **kwargs):
        self._pending = []
        self._response = None
        self.messages_received = []
        self._server_comm = MPI.COMM_SELF if comm is None else comm
        super().__init__(*args, **kwargs)

    def _load_the_model(self):
        self.prob = om.Problem(comm=self._server_comm)
        self.prob.model = self.get_om_group_function_pointer()
        self.prob.setup(mode="rev")
        self.comm = self.prob.model.comm

    def handle(self, message: str = None) -> str:
        self._pending = [message, "shutdown|null"]
        self.run()
        return self._response

    def _parse_incoming_message(self):
        message = self._pending.pop(0)
        if message != "shutdown|null":
            message = self._server_comm.bcast(message, root=0)
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
        raise AssertionError(
            f"Client contacted the server unexpectedly: {message[:40]}"
        )


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
