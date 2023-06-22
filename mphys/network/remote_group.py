import openmdao.api as om
import json, time
import numpy as np

from pbs4py import PBS
from mphys.network.zeromq_server_manager import MPhysZeroMQServerManager

var_naming_dot_replacement = ':'

class RemoteGroup(om.Group):
    def __init__(self,
                 run_server_filename,
                 pbs: PBS,
                 port=5080,
                 acceptable_port_range=[5080,6000],
                 time_estimate_multiplier=2.,
                 reboot_only_on_function_call=True,
                 dump_json=False
                 ):
        super().__init__()
        self.run_server_filename = run_server_filename
        self.pbs = pbs
        self.port = port
        self.acceptable_port_range = acceptable_port_range
        self.time_estimate_multiplier = time_estimate_multiplier
        self.reboot_only_on_function_call = reboot_only_on_function_call
        self.dump_json = dump_json

    def setup(self):
        if self.comm.size>1:
            raise SystemError('Using BatchModel on more than 1 rank is not supported')
        self.add_subsystem('mphys_analysis', RemoteComp(
                                        run_server_filename=self.run_server_filename,
                                        pbs=self.pbs,
                                        port=self.port,
                                        acceptable_port_range=self.acceptable_port_range,
                                        time_estimate_multiplier=self.time_estimate_multiplier,
                                        reboot_only_on_function_call=self.reboot_only_on_function_call,
                                        dump_json=self.dump_json
                                    ), promotes=['*'])

    def stop_server(self):
        # shortcut for stopping server from top level
        self.mphys_analysis.server_manager.stop_server()

class RemoteComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('run_server_filename') # file that will launch MPhysZeroMQServer, with --port arg
        self.options.declare('pbs') # pbs4py object
        self.options.declare('port') # port number for server/client communication
        self.options.declare('acceptable_port_range') # port range to look through if 'port' is currently busy
        self.options.declare('time_estimate_multiplier') # when determining whther to reboot the server, estimate model run time as this times max prior run time
        self.options.declare('reboot_only_on_function_call') # only allows server reboot before function call, not gradient call
                                                             # avoids having to rerun forward solution on next job, but shortens current job time
        self.options.declare('dump_json') # dump input/output json file in client

    def setup(self):
        self.time_estimate_multiplier = self.options['time_estimate_multiplier']
        self.reboot_only_on_function_call = self.options['reboot_only_on_function_call']
        self.dump_json = self.options['dump_json']

        # setup the server
        self.server_manager = MPhysZeroMQServerManager(pbs=self.options['pbs'],
                                                       run_server_filename=self.options['run_server_filename'],
                                                       port=self.options['port'],
                                                       acceptable_port_range=self.options['acceptable_port_range'])

        # for tracking model times, and determining whether to relaunch servers
        self.times_function = np.array([])
        self.times_gradient = np.array([])

        # get baseline model
        print('CLIENT: Running model from setup to get variable sizes', flush=True)
        output_dict = self.evaluate_model(command='initialize')

        # add inputs
        for dv in output_dict['design_vars'].keys():
            self.add_input(dv.replace('.',var_naming_dot_replacement), output_dict['design_vars'][dv]['val'])
            self.add_design_var(dv.replace('.',var_naming_dot_replacement), ref=output_dict['design_vars'][dv]['ref'],
                                                                            lower=output_dict['design_vars'][dv]['lower'],
                                                                            upper=output_dict['design_vars'][dv]['upper'])

        # add outputs
        for obj in output_dict['objective'].keys():
            self.add_output(obj.replace('.',var_naming_dot_replacement), output_dict['objective'][obj]['val'])
            self.add_objective(obj.replace('.',var_naming_dot_replacement), ref=output_dict['objective'][obj]['ref'])
        for con in output_dict['constraints'].keys():
            self.add_output(con.replace('.',var_naming_dot_replacement), output_dict['constraints'][con]['val'])
            if output_dict['constraints'][con]['equals'] is not None: # equality constraint
                self.add_constraint(con.replace('.',var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                 equals=output_dict['constraints'][con]['equals'])
            else:
                if output_dict['constraints'][con]['lower']>-1e20 and output_dict['constraints'][con]['upper']<1e20: # enforce lower and upper bounds
                    self.add_constraint(con.replace('.',var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                     lower=output_dict['constraints'][con]['lower'],
                                                                                     upper=output_dict['constraints'][con]['upper'])
                elif output_dict['constraints'][con]['lower']>-1e20: # enforce lower bound
                    self.add_constraint(con.replace('.',var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                     lower=output_dict['constraints'][con]['lower'])
                else: # enforce upper bound
                    self.add_constraint(con.replace('.',var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                     upper=output_dict['constraints'][con]['upper'])

        # save keys for later
        self.design_vars = output_dict['design_vars'].keys()
        self.objective = output_dict['objective'].keys()
        self.constraints = output_dict['constraints'].keys()

        # declare partials for derivatives
        self.declare_partials('*', '*')

    def compute(self,inputs,outputs):

        input_dict = self._assign_input_dict_for_server(inputs)
        output_dict = self.evaluate_model(input_dict=input_dict, command='evaluate')

        # assign outputs
        for obj in output_dict['objective'].keys():
            outputs[obj.replace('.',var_naming_dot_replacement)] = output_dict['objective'][obj]['val']
        for con in output_dict['constraints'].keys():
            outputs[con.replace('.',var_naming_dot_replacement)] = output_dict['constraints'][con]['val']

    def compute_partials(self, inputs, partials):
        # NOTE: this will not use of and wrt inputs, if given in outer script's compute_totals/check_totals

        input_dict = self._assign_input_dict_for_server(inputs)
        output_dict = self.evaluate_model(input_dict=input_dict, command='evaluate derivatives')

        # assign derivatives
        for obj in output_dict['objective'].keys():
            for dv in output_dict['design_vars'].keys():
                partials[( obj.replace('.',var_naming_dot_replacement), dv.replace('.',var_naming_dot_replacement) )] = output_dict['objective'][obj]['derivatives'][dv]
        for con in output_dict['constraints'].keys():
            for dv in output_dict['design_vars'].keys():
                partials[( con.replace('.',var_naming_dot_replacement), dv.replace('.',var_naming_dot_replacement) )] = output_dict['constraints'][con]['derivatives'][dv]

    def _assign_input_dict_for_server(self, inputs):
        input_dict = {'design_vars': {}}
        for dv in self.design_vars:
            input_dict['design_vars'][dv.replace('.',var_naming_dot_replacement)] = {'val': inputs[dv.replace('.',var_naming_dot_replacement)].tolist()}
        return input_dict

    def _doing_derivative_evaluation(self, command: str):
        return command == 'evaluate derivatives'

    def _is_first_function_evaluation(self):
        return len(self.times_function) == 0

    def _is_first_gradient_evaluation(self):
        return len(self.times_gradient) == 0

    def _need_to_restart_server(self, command: str):
        if self._doing_derivative_evaluation(command):
            if self._is_first_gradient_evaluation() or self.reboot_only_on_function_call:
                return False
            else:
                estimated_model_time = self.time_estimate_multiplier*max(self.times_gradient)

        else:
            if self._is_first_function_evaluation():
                return False
            else:
                if self.reboot_only_on_function_call and not self._is_first_gradient_evaluation():
                    estimated_model_time = self.time_estimate_multiplier*(max(self.times_function)+max(self.times_gradient))
                else:
                    estimated_model_time = self.time_estimate_multiplier*max(self.times_function)
        return not self.server_manager.enough_time_is_remaining(estimated_model_time)

    def _dump_input_json(self, input_dict: dict):
        with open('batch_inputs.json', 'w') as f:
            json.dump(input_dict, f, indent=4)

    def _dump_output_json(self, output_dict: dict):
        with open('batch_outputs.json', 'w') as f:
            json.dump(output_dict, f, indent=4)

    def _send_inputs_to_server(self, input_dict, command: str):
        print('CLIENT: Sending new design to server', flush=True)
        input_str = f"{command}|{str(json.dumps(input_dict))}"
        self.server_manager.socket.send(input_str.encode())

    def _receive_outputs_from_server(self):
        return json.loads(self.server_manager.socket.recv().decode())

    def evaluate_model(self, input_dict=None, command='initialize'):

        need_derivatives = self._doing_derivative_evaluation(command)

        if self._need_to_restart_server(command):
            self.server_manager.stop_server()
            self.server_manager.start_server()

        if self.dump_json:
            self._dump_input_json(input_dict)

        model_start_time = time.time()
        self._send_inputs_to_server(input_dict, command)
        output_dict = self._receive_outputs_from_server()

        if need_derivatives:
            self.times_gradient = np.hstack([self.times_gradient, time.time()-model_start_time])
        else:
            self.times_function = np.hstack([self.times_function, time.time()-model_start_time])

        if self.dump_json:
            self._dump_output_json(output_dict)

        return output_dict
