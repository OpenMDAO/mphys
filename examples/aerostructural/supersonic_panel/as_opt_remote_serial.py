import openmdao.api as om
from as_opt_parallel import write_out_optimization_data
from pbs4py import PBS

from mphys.network.zmq_pbs import RemoteZeroMQComp


def run_check_totals(prob: om.Problem):
    prob.setup(mode="rev")
    om.n2(prob, show_browser=False, outfile="n2.html")
    prob.run_model()
    prob.check_totals(
        step_calc="rel_avg",
        compact_print=True,
        directional=False,
        show_progress=True,
    )
    prob.model.remote.stop_server()


def run_optimization(prob: om.Problem):
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

    sql_file = "optimization_history.sql"
    recorder = om.SqliteRecorder(sql_file)
    prob.driver.add_recorder(recorder)

    prob.setup(mode="rev")
    prob.run_driver()
    prob.model.remote.stop_server()
    prob.cleanup()

    if prob.model.comm.rank == 0:

        write_out_optimization_data(prob, sql_file)

        with open(sql_file, "a") as f:
            f.write("run times, function\n")
            for i in range(len(prob.model.remote.times_function)):
                f.write(f"{prob.model.remote.times_function[i]}\n")
            f.write(" " + "\n")

            f.write("run times, gradient\n")
            for i in range(len(prob.model.remote.times_gradient)):
                f.write(f"{prob.model.remote.times_gradient[i]}\n")
            f.write(" " + "\n")


def main():
    check_totals = False

    pbs = PBS.k4(time=1)
    pbs.mpiexec = "mpirun"
    pbs.requested_number_of_nodes = 1

    prob = om.Problem()
    prob.model.add_subsystem(
        "remote",
        RemoteZeroMQComp(
            run_server_filename="mphys_server.py",  # default server filename
            pbs=pbs,
            additional_server_args="--model_filename as_opt_parallel "
            + "--scenario_name cruise pullup",
        ),  # customizable options for server file
    )

    if check_totals:
        run_check_totals(prob)
    else:
        run_optimization(prob)


if __name__ == "__main__":
    main()
