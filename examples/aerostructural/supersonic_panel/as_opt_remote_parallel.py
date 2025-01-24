import numpy as np
import openmdao.api as om
from pbs4py import PBS

from mphys.network.zmq_pbs import RemoteZeroMQComp

# True=check objective/constraint derivatives, False=run optimization
check_totals = False


# for running scenarios on different servers in parallel
class ParallelRemoteGroup(om.ParallelGroup):
    def initialize(self):
        self.options.declare("num_scenarios")

    def setup(self):
        # NOTE: make sure setup isn't called multiple times, otherwise the first jobs/port forwarding will go unused and you'll have to stop them manually
        for i in range(self.options["num_scenarios"]):

            # initialize pbs4py Launcher
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
            self.add_subsystem(
                f"remote_scenario{i}",
                RemoteZeroMQComp(
                    run_server_filename="mphys_server.py",
                    pbs=pbs_launcher,
                    port=5054 + i * 4,
                    acceptable_port_range=[5054 + i * 4, 5054 + (i + 1) * 4 - 1],
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

        # distributed IVCs
        self.ivc.add_output("mach", [5.0, 3.0])
        self.ivc.add_output("qdyn", [3e4, 1e4])
        self.ivc.add_output("aoa", [3.0, 2.0])

        # add distributed design vars
        self.add_design_var("aoa", lower=-20.0, upper=20.0)

        # add the parallel servers
        self.add_subsystem(
            "multipoint", ParallelRemoteGroup(num_scenarios=2), promotes=["*"]
        )

        # connect distributed IVCs to servers, which are size (2,) and (1,) on client and server sides
        for i in range(2):
            for var in ["mach", "qdyn", "aoa"]:
                self.connect(var, f"remote_scenario{i}.{var}", src_indices=[i])

        # add CL and stress constraints
        min_CL = [0.15, 0.45]
        for i in range(2):
            self.add_constraint(
                f"aerostructural{i}:C_L",
                lower=min_CL[i],
                ref=0.1,
                parallel_deriv_color="lift_cons",
            )
            self.add_constraint(
                f"aerostructural{i}:func_struct",
                upper=1.0,
                parallel_deriv_color="struct_cons",
            )

        # add objective
        self.add_objective("aerostructural0:mass", ref=0.01)


# add remote component to the model
prob = om.Problem()
prob.model = TopLevelGroup()

if check_totals:
    prob.setup(mode="rev")
    om.n2(prob, show_browser=False, outfile="n2.html")
    prob.run_model()
    prob.check_totals(
        step_calc="rel_avg", compact_print=True, directional=False, show_progress=True
    )

else:

    # setup optimization driver
    prob.driver = om.ScipyOptimizeDriver(
        debug_print=["nl_cons", "objs", "desvars", "totals"]
    )
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-5
    prob.driver.options["disp"] = True
    prob.driver.options["maxiter"] = 300

    # add optimization recorder
    prob.driver.recording_options["record_objectives"] = True
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_derivatives"] = True

    recorder = om.SqliteRecorder("optimization_history_parallel.sql")
    prob.driver.add_recorder(recorder)

    # run the optimization
    prob.setup(mode="rev")
    prob.run_driver()
    prob.cleanup()

    # write out data
    if prob.model.comm.rank == 0:
        cr = om.CaseReader(
            f"{prob.get_outputs_dir()}/optimization_history_parallel.sql"
        )
        driver_cases = cr.list_cases("driver")

        case = cr.get_case(0)
        cons = case.get_constraints()
        dvs = case.get_design_vars()
        objs = case.get_objectives()

        with open("optimization_history.dat", "w+") as f:

            for i, k in enumerate(objs.keys()):
                f.write("objective: " + k + "\n")
                for j, case_id in enumerate(driver_cases):
                    f.write(
                        str(j)
                        + " "
                        + str(cr.get_case(case_id).get_objectives(scaled=False)[k][0])
                        + "\n"
                    )
                f.write(" " + "\n")

            for i, k in enumerate(cons.keys()):
                f.write("constraint: " + k + "\n")
                for j, case_id in enumerate(driver_cases):
                    f.write(
                        str(j)
                        + " "
                        + " ".join(
                            map(
                                str,
                                cr.get_case(case_id).get_constraints(scaled=False)[k],
                            )
                        )
                        + "\n"
                    )
                f.write(" " + "\n")

            for i, k in enumerate(dvs.keys()):
                f.write("DV: " + k + "\n")
                for j, case_id in enumerate(driver_cases):
                    f.write(
                        str(j)
                        + " "
                        + " ".join(
                            map(
                                str,
                                cr.get_case(case_id).get_design_vars(scaled=False)[k],
                            )
                        )
                        + "\n"
                    )
                f.write(" " + "\n")

# shutdown the servers
prob.model.multipoint.remote_scenario0.stop_server()
prob.model.multipoint.remote_scenario1.stop_server()
