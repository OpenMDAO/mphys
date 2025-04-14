import openmdao.api as om
from as_opt_parallel import run_check_totals, run_optimization
from pbs4py import PBS

from mphys.network.zmq_pbs import RemoteZeroMQComp


# for running scenarios on different servers in parallel
class ParallelRemoteGroup(om.ParallelGroup):
    def initialize(self):
        self.options.declare("num_scenarios")

    def setup(self):
        # NOTE: make sure setup isn't called multiple times, otherwise the first jobs/port forwarding will go unused and you'll have to stop them manually
        for i in range(self.options["num_scenarios"]):

            pbs_launcher = PBS.k4(time=1)
            pbs_launcher.mpiexec = "mpirun"
            pbs_launcher.requested_number_of_nodes = 1

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

            # add the remote server component
            start_port = 5054 + i * 4
            end_port = 5054 + (i + 1) * 4 - 1
            self.add_subsystem(
                f"remote_scenario{i}",
                RemoteZeroMQComp(
                    run_server_filename="mphys_server.py",
                    pbs=pbs_launcher,
                    port=start_port,
                    acceptable_port_range=[start_port, end_port],
                    dump_separate_json=True,
                    additional_remote_inputs=["mach", "qdyn", "aoa"],
                    additional_remote_outputs=additional_remote_outputs,
                    additional_server_args=f"--model_filename run --scenario_name aerostructural{i}",
                ),
                promotes_inputs=[
                    "geometry_morph_param",
                    "dv_struct",
                ],  # non-distributed IVCs
                promotes_outputs=["*"],
            )


class TopLevelGroup(om.Group):
    def setup(self):
        # IVCs that feed into both parallel groups
        self.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])

        self.ivc.add_output("mach", [5.0, 3.0])
        self.ivc.add_output("qdyn", [3e4, 1e4])
        self.ivc.add_output("aoa", [3.0, 2.0])

        self.add_design_var("aoa", lower=-20.0, upper=20.0)

        self.add_subsystem(
            "multipoint", ParallelRemoteGroup(num_scenarios=2), promotes=["*"]
        )

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
