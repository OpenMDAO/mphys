from mphys.network.zmq_pbs import (MPhysZeroMQServer,
                                   get_default_zmq_pbs_argparser)


class GetModel:
    def __init__(self, scenario_name: str, model_filename: str):
        self.scenario_name = scenario_name
        self.model_filename = model_filename
    def __call__(self):
        return __import__(self.model_filename).get_model(self.scenario_name)

def run_server(args):
    get_model = GetModel(args.scenario_name, args.model_filename)
    server = MPhysZeroMQServer(args.port,
                               get_om_group_function_pointer=get_model,
                               ignore_setup_warnings=True,
                               ignore_runtime_warnings=True,
                               rerun_initial_design=True)
    server.run()

if __name__ == "__main__":
    # argparse with port number input
    parser = get_default_zmq_pbs_argparser()

    # add some customizable options
    parser.add_argument('--model_filename', type=str, default='run',
                        help='filename (excluding .py) containing the get_model function to set as get_om_group_function_pointer')
    parser.add_argument('--scenario_name', nargs='+', type=str, default=['cruise'],
                        help='scenario name to feed into get_om_group_function_pointer')

    args = parser.parse_args()
    run_server(args)
