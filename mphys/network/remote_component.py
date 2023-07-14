import openmdao.api as om
import json, time
import numpy as np

class RemoteComp(om.ExplicitComponent):
    """
    A component used for network communication between top-level OpenMDAO
    problem and remote problem evaluated on an HPC job. Serves as the
    top-level component on the client side.

    To make a particular derived class, implement the _setup_server_manager,
    _send_inputs_to_server, and _receive_outputs_from_server functions.
    """
    def stop_server(self):
        # shortcut for stopping server from top level
        self.server_manager.stop_server()

    def initialize(self):
        self.options.declare('run_server_filename', desc="python file that will launch the Server class")
        self.options.declare('time_estimate_multiplier', default=2.0, desc="when determining whether to reboot the server, estimate model run time as this times max prior run time")
        self.options.declare('reboot_only_on_function_call', default=True, desc="only allows server reboot before function call, not gradient call. "
                                                                                +"Avoids having to rerun forward solution on next job, but shortens current job time")
        self.options.declare('dump_json', default=False, desc="dump input/output json file in client")
        self.options.declare('var_naming_dot_replacement', default=":", desc="what to replace '.' within dv/response name trees")
        self.options.declare('additional_remote_inputs', default=[], types=list, desc="additional inputs not defined as design vars in the remote component")
        self.options.declare('additional_remote_outputs', default=[], types=list, desc="additional outputs not defined as objective/constraints in the remote component")

    def setup(self):
        if self.comm.size>1:
            raise SystemError('Using Remote Component on more than 1 rank is not supported')
        self.time_estimate_multiplier = self.options['time_estimate_multiplier']
        self.reboot_only_on_function_call = self.options['reboot_only_on_function_call']
        self.dump_json = self.options['dump_json']
        self.var_naming_dot_replacement = self.options['var_naming_dot_replacement']
        self.additional_remote_inputs = self.options['additional_remote_inputs']
        self.additional_remote_outputs = self.options['additional_remote_outputs']

        self._setup_server_manager()

        # for tracking model times, and determining whether to relaunch servers
        self.times_function = np.array([])
        self.times_gradient = np.array([])

        # get baseline model
        print('CLIENT: Running model from setup to get design problem info', flush=True)
        output_dict = self.evaluate_model(command='initialize',
                                          remote_input_dict={'additional_inputs': self.additional_remote_inputs,
                                                             'additional_outputs': self.additional_remote_outputs})

        #TODO support scalers, adders.
        self._add_design_inputs_from_baseline_model(output_dict)
        self._add_objectives_from_baseline_model(output_dict)
        self._add_constraints_from_baseline_model(output_dict)

        self._add_additional_inputs_from_baseline_model(output_dict)
        self._add_additional_outputs_from_baseline_model(output_dict)

        self.declare_partials('*', '*')

    def compute(self,inputs,outputs):
        input_dict = self._create_input_dict_for_server(inputs)
        remote_dict = self.evaluate_model(remote_input_dict=input_dict, command='evaluate')

        self._assign_objectives_from_remote_output(remote_dict, outputs)
        self._assign_constraints_from_remote_output(remote_dict, outputs)
        self._assign_additional_outputs_from_remote_output(remote_dict, outputs)

    def compute_partials(self, inputs, partials):
        # NOTE: this will not use of and wrt inputs, if given in outer script's compute_totals/check_totals

        input_dict = self._create_input_dict_for_server(inputs)
        remote_dict = self.evaluate_model(remote_input_dict=input_dict, command='evaluate derivatives')

        self._assign_objective_partials_from_remote_output(remote_dict, partials)
        self._assign_constraint_partials_from_remote_output(remote_dict, partials)
        self._assign_additional_partials_from_remote_output(remote_dict, partials)

    def evaluate_model(self, remote_input_dict=None, command='initialize'):
        if self._need_to_restart_server(command):
            self.server_manager.stop_server()
            self.server_manager.start_server()

        if self.dump_json:
            self._dump_input_json(remote_input_dict)

        model_start_time = time.time()
        self._send_inputs_to_server(remote_input_dict, command)
        remote_output_dict = self._receive_outputs_from_server()

        model_time_elapsed = time.time() - model_start_time
        if self._doing_derivative_evaluation(command):
            self.times_gradient = np.hstack([self.times_gradient, model_time_elapsed])
        else:
            self.times_function = np.hstack([self.times_function, model_time_elapsed])

        if self.dump_json:
            self._dump_output_json(remote_output_dict)

        return remote_output_dict

    def _assign_objective_partials_from_remote_output(self, remote_dict, partials):
        for obj in remote_dict['objective'].keys():
            for dv in remote_dict['design_vars'].keys():
                partials[( obj.replace('.',self.var_naming_dot_replacement), dv.replace('.',self.var_naming_dot_replacement) )] = remote_dict['objective'][obj]['derivatives'][dv]
            for inp in remote_dict['additional_inputs'].keys():
                partials[( obj.replace('.',self.var_naming_dot_replacement), inp.replace('.',self.var_naming_dot_replacement) )] = remote_dict['objective'][obj]['derivatives'][inp]

    def _assign_constraint_partials_from_remote_output(self, remote_dict, partials):
        for con in remote_dict['constraints'].keys():
            for dv in remote_dict['design_vars'].keys():
                partials[( con.replace('.',self.var_naming_dot_replacement), dv.replace('.',self.var_naming_dot_replacement) )] = remote_dict['constraints'][con]['derivatives'][dv]
            for inp in remote_dict['additional_inputs'].keys():
                partials[( con.replace('.',self.var_naming_dot_replacement), inp.replace('.',self.var_naming_dot_replacement) )] = remote_dict['constraints'][con]['derivatives'][inp]

    def _assign_additional_partials_from_remote_output(self, remote_dict, partials):
        for output in remote_dict['additional_outputs'].keys():
            for dv in remote_dict['design_vars'].keys():
                partials[( output.replace('.',self.var_naming_dot_replacement), dv.replace('.',self.var_naming_dot_replacement) )] = remote_dict['additional_outputs'][output]['derivatives'][dv]
            for inp in remote_dict['additional_inputs'].keys():
                partials[( output.replace('.',self.var_naming_dot_replacement), inp.replace('.',self.var_naming_dot_replacement) )] = remote_dict['additional_outputs'][output]['derivatives'][inp]

    def _create_input_dict_for_server(self, inputs):
        input_dict = {'design_vars': {}, 'additional_inputs': {}, 'additional_outputs': self.additional_remote_outputs}
        for dv in self.design_var_keys:
            input_dict['design_vars'][dv.replace('.',self.var_naming_dot_replacement)] = {'val': inputs[dv.replace('.',self.var_naming_dot_replacement)].tolist()}
        for input in self.additional_remote_inputs:
            input_dict['additional_inputs'][input] = {'val': inputs[input.replace('.',self.var_naming_dot_replacement)].tolist()}
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

    def _dump_input_json(self, remote_input_dict: dict):
        with open('remote_inputs.json', 'w') as f:
            json.dump(remote_input_dict, f, indent=4)

    def _dump_output_json(self, remote_output_dict: dict):
        with open('remote_outputs.json', 'w') as f:
            json.dump(remote_output_dict, f, indent=4)

    def _add_design_inputs_from_baseline_model(self, output_dict):
        self.design_var_keys = output_dict['design_vars'].keys()
        for dv in self.design_var_keys:
            self.add_input(dv.replace('.',self.var_naming_dot_replacement), output_dict['design_vars'][dv]['val'])
            self.add_design_var(dv.replace('.',self.var_naming_dot_replacement), ref=output_dict['design_vars'][dv]['ref'],
                                                                                 lower=output_dict['design_vars'][dv]['lower'],
                                                                                 upper=output_dict['design_vars'][dv]['upper'])

    def _add_additional_inputs_from_baseline_model(self, output_dict):
        self.additional_remote_inputs = list(output_dict['additional_inputs'].keys())
        for input in self.additional_remote_inputs:
            self.add_input(input.replace('.',self.var_naming_dot_replacement), output_dict['additional_inputs'][input]['val'])

    def _add_additional_outputs_from_baseline_model(self, output_dict):
        self.additional_remote_outputs = list(output_dict['additional_outputs'].keys())
        for output in self.additional_remote_outputs:
            self.add_output(output.replace('.',self.var_naming_dot_replacement), output_dict['additional_outputs'][output]['val'])

    def _add_objectives_from_baseline_model(self, output_dict):
        for obj in output_dict['objective'].keys():
            self.add_output(obj.replace('.',self.var_naming_dot_replacement), output_dict['objective'][obj]['val'])
            self.add_objective(obj.replace('.',self.var_naming_dot_replacement), ref=output_dict['objective'][obj]['ref'])

    def _add_constraints_from_baseline_model(self, output_dict):
        for con in output_dict['constraints'].keys():
            self.add_output(con.replace('.',self.var_naming_dot_replacement), output_dict['constraints'][con]['val'])
            if output_dict['constraints'][con]['equals'] is not None: # equality constraint
                self.add_constraint(con.replace('.',self.var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                      equals=output_dict['constraints'][con]['equals'])
            else:
                if output_dict['constraints'][con]['lower']>-1e20 and output_dict['constraints'][con]['upper']<1e20: # enforce lower and upper bounds
                    self.add_constraint(con.replace('.',self.var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                          lower=output_dict['constraints'][con]['lower'],
                                                                                          upper=output_dict['constraints'][con]['upper'])
                elif output_dict['constraints'][con]['lower']>-1e20: # enforce lower bound
                    self.add_constraint(con.replace('.',self.var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                          lower=output_dict['constraints'][con]['lower'])
                else: # enforce upper bound
                    self.add_constraint(con.replace('.',self.var_naming_dot_replacement), ref=output_dict['constraints'][con]['ref'],
                                                                                          upper=output_dict['constraints'][con]['upper'])

    def _assign_objectives_from_remote_output(self, remote_dict, outputs):
        for obj in remote_dict['objective'].keys():
            outputs[obj.replace('.',self.var_naming_dot_replacement)] = remote_dict['objective'][obj]['val']

    def _assign_constraints_from_remote_output(self, remote_dict, outputs):
        for con in remote_dict['constraints'].keys():
            outputs[con.replace('.',self.var_naming_dot_replacement)] = remote_dict['constraints'][con]['val']

    def _assign_additional_outputs_from_remote_output(self, remote_dict, outputs):
        for output in remote_dict['additional_outputs'].keys():
            outputs[output.replace('.',self.var_naming_dot_replacement)] = remote_dict['additional_outputs'][output]['val']

    def _send_inputs_to_server(self, remote_input_dict, command: str):
        raise NotImplementedError

    def _receive_outputs_from_server(self):
        raise NotImplementedError

    def _setup_server_manager(self):
        raise NotImplementedError
