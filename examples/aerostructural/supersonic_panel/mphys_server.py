from mphys.network.zeromq_server import MPhysZeroMQServer
import argparse

def get_server_om_group():
    from as_opt_parallel import Model
    return Model()

def run_server(port):
    server = MPhysZeroMQServer(port,
                               get_om_group_function_pointer=get_server_om_group,
                               ignore_setup_warnings=True,
                               ignore_runtime_warnings=True,
                               rerun_initial_design=True)
    server.run()

if __name__ == "__main__":
    # called from MPhysZeroMQServerManager. Must accept --port as argument
    parser = argparse.ArgumentParser('Python script for launching mphys analysis server',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, help='tcp port number for zeromq socket')
    args = vars(parser.parse_args())
    run_server(args['port'])
