from mphys.network.zmq_pbs import MPhysZeroMQServer, get_default_zmq_pbs_argparser

def run_server(args):
    # some options to hand to your get_model function
    options = {'scenario_name': args.scenario_name}

    server = MPhysZeroMQServer(args.port,
                               get_om_group_function_pointer=__import__(args.filename).get_model,
                               function_pointer_options_dict=options,
                               ignore_setup_warnings=True,
                               ignore_runtime_warnings=True,
                               rerun_initial_design=True)
    server.run()

if __name__ == "__main__":
    # argparse with port number input
    parser = get_default_zmq_pbs_argparser()

    # add some customizable options
    parser.add_argument('--filename', type=str, default='run',
                        help='filename (excluding .py) containing the get_model function to set as get_om_group_function_pointer')
    parser.add_argument('--scenario_name', nargs='+', type=str, default=['cruise'],
                        help='scenario name to feed into get_om_group_function_pointer')

    args = parser.parse_args()
    run_server(args)
