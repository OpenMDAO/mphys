import openmdao.api as om
from mphys.network.zmq_pbs import RemoteZeroMQComp
from pbs4py import PBS

check_totals = False # True=check objective/constraint derivatives, False=run optimization

# initialize pbs4py
pbs = PBS.k4(time=1)
pbs.mpiexec = 'mpirun'
pbs.requested_number_of_nodes = 1

# add remote component to the model
prob = om.Problem()
prob.model.add_subsystem('remote',
                        RemoteZeroMQComp(
                            run_server_filename='mphys_server.py', # default server filename
                            pbs=pbs,
                            additional_server_args='--model_filename as_opt_parallel ' +
                                                   '--scenario_name cruise pullup') # customizable options for server file
                        )

if check_totals:
    prob.setup(mode='rev')
    om.n2(prob, show_browser=False, outfile='n2.html')
    prob.run_model()
    prob.check_totals(step_calc='rel_avg',
                        compact_print=True,
                        directional=False,
                        show_progress=True)
    prob.model.remote.stop_server()

else:

    # setup optimization driver
    prob.driver = om.ScipyOptimizeDriver(debug_print=['nl_cons','objs','desvars','totals'])
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-5
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 300

    # add optimization recorder
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['record_derivatives'] = True

    recorder = om.SqliteRecorder("optimization_history.sql")
    prob.driver.add_recorder(recorder)

    # run the optimization
    prob.setup(mode='rev')
    prob.run_driver()
    prob.model.remote.stop_server()
    prob.cleanup()

    # write out data
    if prob.model.comm.rank==0:
        cr = om.CaseReader("optimization_history.sql")
        driver_cases = cr.list_cases('driver')

        case = cr.get_case(0)
        cons = case.get_constraints()
        dvs = case.get_design_vars()
        objs = case.get_objectives()

        with open("optimization_history.dat","w+") as f:

            for i, k in enumerate(objs.keys()):
                f.write('objective: ' + k + '\n')
                for j, case_id in enumerate(driver_cases):
                    f.write(str(j) + ' ' + str(cr.get_case(case_id).get_objectives(scaled=False)[k][0]) + '\n')
                f.write(' ' + '\n')

            for i, k in enumerate(cons.keys()):
                f.write('constraint: ' + k + '\n')
                for j, case_id in enumerate(driver_cases):
                    f.write(str(j) + ' ' + ' '.join(map(str,cr.get_case(case_id).get_constraints(scaled=False)[k])) + '\n')
                f.write(' ' + '\n')

            for i, k in enumerate(dvs.keys()):
                f.write('DV: ' + k + '\n')
                for j, case_id in enumerate(driver_cases):
                    f.write(str(j) + ' ' + ' '.join(map(str,cr.get_case(case_id).get_design_vars(scaled=False)[k])) + '\n')
                f.write(' ' + '\n')

            f.write('run times, function\n')
            for i in range(len(prob.model.remote.times_function)):
                f.write(f'{prob.model.remote.times_function[i]}\n')
            f.write(' ' + '\n')

            f.write('run times, gradient\n')
            for i in range(len(prob.model.remote.times_gradient)):
                f.write(f'{prob.model.remote.times_gradient[i]}\n')
            f.write(' ' + '\n')
