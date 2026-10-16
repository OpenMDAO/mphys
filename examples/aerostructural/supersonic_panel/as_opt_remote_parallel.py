import os
import socket

import openmdao.api as om
from as_opt_parallel import run_check_totals, run_optimization
from pbs4py import PBS

from mphys.network.zmq_pbs import RemoteZeroMQComp


# for running scenarios on different servers in parallel
class ParallelRemoteGroup(om.ParallelGroup):
    def initialize(self):
        self.options.declare("num_scenarios")
        # the remote components are created once and re-added on every setup call: a new
        # component per setup would launch a new server job and ssh tunnel each time
        self._remote_components = {}

    def setup(self):
        for i in range(self.options["num_scenarios"]):
            name = f"remote_scenario{i}"
            if name not in self._remote_components:
                self._remote_components[name] = self._create_remote_component(i)
            self.add_subsystem(
                name,
                self._remote_components[name],
                promotes_inputs=[
                    "geometry_morph_param",
                    "dv_struct",
                ],  # non-distributed IVCs
                promotes_outputs=["*"],
            )

    def _create_remote_component(self, i):
        pbs_launcher, hpc = self._get_pbs_launcher()

        # output functions of interest, which aren't already added as objective/constraints on server side
        if i == 0:
            additional_remote_outputs = [
                f"aerostructural{i}.mass",
                f"aerostructural{i}.C_L",
                f"aerostructural{i}.func_struct",
            ]
        else:  # exclude mass (which comes from first scenario), otherwise mass derivatives will be computed needlessly
            additional_remote_outputs = [
                f"aerostructural{i}.C_L",
                f"aerostructural{i}.func_struct",
            ]

        start_port = 5054 + i * 4
        end_port = 5054 + (i + 1) * 4 - 1
        return RemoteZeroMQComp(
            run_server_filename="mphys_server.py",
            pbs=pbs_launcher,
            port=start_port,
            acceptable_port_range=[start_port, end_port],
            forward_through_frontend=True if hpc == "nas" else False,
            dump_separate_json=True,
            additional_remote_inputs=["mach", "qdyn", "aoa"],
            additional_remote_outputs=additional_remote_outputs,
            additional_server_args=f"--model_filename run --scenario_name aerostructural{i}",
        )

    def _get_pbs_launcher(self):

        # get hostname
        if os.environ.get("PBS_O_HOST") is not None:  # running from HPC job
            host = os.environ.get("PBS_O_HOST")
        else:  # running from login node
            host = socket.gethostname()

        # check if using nas or k
        if host.startswith("k4-li"):
            hpc = "k"
        elif host.startswith("pfe"):
            hpc = "nas"
        else:
            raise ValueError(
                f"Unable to determine if running from NAS or K based on hostname '{host}'"
            )

        if hpc == "k":
            pbs_launcher = PBS.k4(
                profile_filename="~/.bashrc",
                requested_number_of_nodes=1,
                time=1,
            )
        elif hpc == "nas":
            pbs_launcher = PBS.nas(
                profile_filename="~/.bashrc",
                requested_number_of_nodes=1,
                time=1,
                # group_list=None, # add group list here
                proc_type="bro",
            )

        return pbs_launcher, hpc


class TopLevelGroup(om.Group):
    def initialize(self):
        # created once, so that the remote components (and their servers) persist across setup calls
        self.multipoint_group = ParallelRemoteGroup(num_scenarios=2)

    def setup(self):
        # IVCs that feed into both parallel groups
        self.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])

        self.ivc.add_output("mach", [5.0, 3.0])
        self.ivc.add_output("qdyn", [3e4, 1e4])
        self.ivc.add_output("aoa", [3.0, 2.0])

        self.add_design_var("aoa", lower=-20.0, upper=20.0)

        self.add_subsystem("multipoint", self.multipoint_group, promotes=["*"])

        for i in range(2):
            # connect distributed IVCs to servers, which are size (2,) and (1,) on client and server sides
            for var in ["mach", "qdyn", "aoa"]:
                self.connect(var, f"remote_scenario{i}.{var}", src_indices=[i])

            # stress constriant
            self.add_constraint(
                f"aerostructural{i}:func_struct",
                upper=1.0,
                parallel_deriv_color="struct_cons",
            )

        min_CLs = [0.15, 0.45]
        for i, min_CL in enumerate(min_CLs):
            self.add_constraint(
                f"aerostructural{i}:C_L",
                lower=min_CL,
                ref=0.1,
                parallel_deriv_color="lift_cons",
            )

        self.add_objective("aerostructural0:mass", ref=0.01)


def main():
    check_totals = False

    prob = om.Problem()
    prob.model = TopLevelGroup()

    if check_totals:
        run_check_totals(prob)
    else:
        run_optimization(prob)

    # shutdown the servers
    prob.model.multipoint.remote_scenario0.stop_server()
    prob.model.multipoint.remote_scenario1.stop_server()


if __name__ == "__main__":
    main()
