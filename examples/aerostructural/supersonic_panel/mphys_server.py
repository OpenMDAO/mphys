from mphys.network.zmq_pbs import MPhysZeroMQServer, get_default_zmq_pbs_argparser
from as_opt_parallel import get_model

def run_server(port):
    server = MPhysZeroMQServer(port,
                               get_om_group_function_pointer=get_model,
                               ignore_setup_warnings=True,
                               ignore_runtime_warnings=True,
                               rerun_initial_design=True)
    server.run()

if __name__ == "__main__":
    parser = get_default_zmq_pbs_argparser()
    args = parser.parse_args()
    run_server(args.port)
