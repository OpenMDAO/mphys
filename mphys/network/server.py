import openmdao.api as om
import warnings

class Server:
    """
    A class that serves as an OpenMDAO model analysis server. Launched
    by a server run file by the ServerManager and runs on an HPC job,
    awaiting design variables to evaluate and sending back resulting
    function or derivative information.

    To make a particular derived class, implement the _parse_incoming_message
    and _send_outputs_to_client functions.

    Parameters
    ----------
    get_om_group_function_pointer : function pointer
        Pointer to the OpenMDAO/MPhys group to evaluate on the server
    ignore_setup_warnings : bool
        Whether to ignore OpenMDAO setup warnings
    ignore_runtime_warnings : bool
        Whether to ignore OpenMDAO runtime warnings
    rerun_initial_design : bool
        Whether to evaluate the baseline design upon starup
    """
    def __init__(self, get_om_group_function_pointer,
                 ignore_setup_warnings = False,
                 ignore_runtime_warnings = False,
                 rerun_initial_design = False):

        self.get_om_group_function_pointer = get_om_group_function_pointer
        self.ignore_setup_warnings = ignore_setup_warnings
        self.ignore_runtime_warnings = ignore_runtime_warnings
        self.rerun_initial_design = rerun_initial_design

        self.current_design_has_been_evaluated = False
        self.current_derivatives_have_been_evaluated = False
        self.derivatives = None
        self.additional_inputs = None
        self.additional_outputs = None
        self.design_counter = 0 # more debugging info for client side json dumping

        self._load_the_model()

    def _parse_incoming_message(self):
        raise NotImplementedError

    def _send_outputs_to_client(self):
        raise NotImplementedError

    def _load_the_model(self):
        self.prob = om.Problem()
        self.prob.model = self.get_om_group_function_pointer()
        if self.ignore_setup_warnings:
            with warnings.catch_warnings(record=True) as w:
                self.prob.setup(mode='rev')
        else:
            self.prob.setup(mode='rev')
        self.rank = self.prob.model.comm.rank

        # temporary fix for MELD initialization issue
        if self.rerun_initial_design:
            if self.rank==0:
                print('SERVER: Evaluating baseline design', flush=True)
            self._run_model()

    def _run_model(self):
        if self.ignore_runtime_warnings:
            with warnings.catch_warnings(record=True) as w:
                self.prob.run_model()
        else:
            self.prob.run_model()
        self.current_design_has_been_evaluated = True
        self.derivatives = None
        self.design_counter += 1

    def _compute_totals(self):
        of, wrt = self._get_derivative_inputs_outputs()
        if self.ignore_runtime_warnings:
            with warnings.catch_warnings(record=True) as w:
                self.derivatives = self.prob.compute_totals(of=of, wrt=wrt)
        else:
            self.derivatives = self.prob.compute_totals(of=of, wrt=wrt)
        self.current_derivatives_have_been_evaluated = True

    def _get_derivative_inputs_outputs(self):
        of = []
        for r in self.prob.model._responses.keys():
            of += [self.prob.model._responses[r]['source']]
        of += self.additional_outputs

        wrt = []
        for dv in self.prob.model._design_vars.keys():
            wrt += [self.prob.model._design_vars[dv]['source']]
        wrt += self.additional_inputs

        return of, wrt

    def _gather_design_inputs_from_om_problem(self, remote_output_dict = {}):
        design_vars = self.prob.model._design_vars
        remote_output_dict['design_vars'] = {}
        for dv in design_vars.keys():
            remote_output_dict['design_vars'][dv] = {'val': self.prob.get_val(dv),
                                                     'ref': design_vars[dv]['ref'],
                                                     'ref0': design_vars[dv]['ref0'],
                                                     'lower': design_vars[dv]['lower'],
                                                     'upper': design_vars[dv]['upper'],
                                                     'units': design_vars[dv]['units']}
            remote_output_dict['design_vars'][dv] = self._set_reference_vals(remote_output_dict['design_vars'][dv], design_vars[dv])
            remote_output_dict['design_vars'][dv] = self._apply_reference_vals_to_desvar_bounds(remote_output_dict['design_vars'][dv])

            # convert to lists for json input/output
            for key in remote_output_dict['design_vars'][dv].keys():
                if hasattr(remote_output_dict['design_vars'][dv][key], 'tolist'):
                    remote_output_dict['design_vars'][dv][key] = remote_output_dict['design_vars'][dv][key].tolist()
        return remote_output_dict

    def _gather_additional_inputs_from_om_problem(self, remote_output_dict = {}):
        remote_output_dict['additional_inputs'] = {}
        for input in self.additional_inputs:
            remote_output_dict['additional_inputs'][input] = {'val': self.prob.get_val(input)}
            if hasattr(remote_output_dict['additional_inputs'][input]['val'], 'tolist'):
                remote_output_dict['additional_inputs'][input]['val'] = remote_output_dict['additional_inputs'][input]['val'].tolist()
        return remote_output_dict

    def _gather_design_outputs_from_om_problem(self, remote_output_dict = {}):
        responses = self.prob.model._responses
        remote_output_dict.update({'objective':{}, 'constraints':{}})
        for r in responses.keys():

            if responses[r]['type']=='obj':
                response_type = 'objective'
            elif responses[r]['type']=='con':
                response_type = 'constraints'

            remote_output_dict[response_type][r] = {'val': self.prob.get_val(r, get_remote=True),
                                                    'ref': responses[r]['ref'],
                                                    'ref0': responses[r]['ref0']}
            remote_output_dict[response_type][r] = self._set_reference_vals(remote_output_dict[response_type][r], responses[r])

            if response_type=='constraints': # get constraint bounds
                remote_output_dict[response_type][r].update({'lower': responses[r]['lower'],
                                                             'upper': responses[r]['upper'],
                                                             'equals': responses[r]['equals']})
                remote_output_dict[response_type][r] = self._apply_reference_vals_to_constraint_bounds(remote_output_dict[response_type][r])

            # convert to lists for json input/output
            for key in remote_output_dict[response_type][r].keys():
                if hasattr(remote_output_dict[response_type][r][key], 'tolist'):
                    remote_output_dict[response_type][r][key] = remote_output_dict[response_type][r][key].tolist()
        return remote_output_dict

    def _set_reference_vals(self, remote_dict, om_dict):
        if remote_dict['ref'] is not None or remote_dict['ref0'] is not None: # using ref/ref0
            remote_dict.update({'scaler': None,
                                'adder': None})
            if remote_dict['ref'] is None:
                remote_dict['ref'] = 1.0
            if remote_dict['ref0'] is None:
                remote_dict['ref0'] = 0.0
        else: # using adder/scaler
            remote_dict.update({'scaler': om_dict['scaler'],
                                'adder': om_dict['adder']})
            if remote_dict['scaler'] is None:
                remote_dict['scaler'] = 1.0
            if remote_dict['adder'] is None:
                remote_dict['adder'] = 0.0
        return remote_dict

    def _apply_reference_vals_to_desvar_bounds(self, desvar_dict):
        if desvar_dict['adder'] is None and desvar_dict['scaler'] is None: # using ref/ref0
            desvar_dict['lower'] = desvar_dict['lower']*(desvar_dict['ref']-desvar_dict['ref0']) + desvar_dict['ref0']
            desvar_dict['upper'] = desvar_dict['upper']*(desvar_dict['ref']-desvar_dict['ref0']) + desvar_dict['ref0']
        else: # using adder/scaler
            desvar_dict['lower'] = desvar_dict['lower']/desvar_dict['scaler'] - desvar_dict['adder']
            desvar_dict['upper'] = desvar_dict['upper']/desvar_dict['scaler'] - desvar_dict['adder']
        return desvar_dict

    def _apply_reference_vals_to_constraint_bounds(self, constraint_dict):
        if constraint_dict['adder'] is None and constraint_dict['scaler'] is None: # using ref/ref0
            if constraint_dict['equals'] is not None: # equality constraint
                constraint_dict['equals'] = constraint_dict['equals']*(constraint_dict['ref']-constraint_dict['ref0']) + constraint_dict['ref0']
            else:
                if constraint_dict['lower']>-1e20:
                    constraint_dict['lower'] = constraint_dict['lower']*(constraint_dict['ref']-constraint_dict['ref0']) + constraint_dict['ref0']
                if constraint_dict['upper']<1e20:
                    constraint_dict['upper'] = constraint_dict['upper']*(constraint_dict['ref']-constraint_dict['ref0']) + constraint_dict['ref0']
        else: # using adder/scaler
            if constraint_dict['equals'] is not None: # equality constraint
                constraint_dict['equals'] = constraint_dict['equals']/constraint_dict['scaler'] - constraint_dict['adder']
            else:
                if constraint_dict['lower']>-1e20:
                    constraint_dict['lower'] = constraint_dict['lower']/constraint_dict['scaler'] - constraint_dict['adder']
                if constraint_dict['upper']<1e20:
                    constraint_dict['upper'] = constraint_dict['upper']/constraint_dict['scaler'] - constraint_dict['adder']
        return constraint_dict

    def _gather_additional_outputs_from_om_problem(self, remote_output_dict = {}):
        remote_output_dict['additional_outputs'] = {}
        for output in self.additional_outputs:
            remote_output_dict['additional_outputs'][output] = {'val': self.prob.get_val(output, get_remote=True)}
            if hasattr(remote_output_dict['additional_outputs'][output]['val'], 'tolist'):
                remote_output_dict['additional_outputs'][output]['val'] = remote_output_dict['additional_outputs'][output]['val'].tolist()
        return remote_output_dict

    def _gather_design_derivatives_from_om_problem(self, remote_output_dict):
        design_vars = self.prob.model._design_vars
        responses = self.prob.model._responses
        for r in responses.keys():

            if responses[r]['type']=='obj':
                response_type = 'objective'
            elif responses[r]['type']=='con':
                response_type = 'constraints'

            remote_output_dict[response_type][r]['derivatives'] = {}
            for dv in design_vars.keys():
                deriv = self.derivatives[(responses[r]['source'], design_vars[dv]['source'])]
                if hasattr(deriv,'tolist'):
                    deriv = deriv.tolist()
                remote_output_dict[response_type][r]['derivatives'][dv] = deriv
        return remote_output_dict

    def _gather_additional_output_derivatives_from_om_problem(self, remote_output_dict):
        for output in self.additional_outputs:
            remote_output_dict['additional_outputs'][output]['derivatives'] = {}

            # wrt design vars
            for dv in self.prob.model._design_vars.keys():
                deriv = self.derivatives[( output , self.prob.model._design_vars[dv]['source'] )]
                if hasattr(deriv,'tolist'):
                    deriv = deriv.tolist()
                remote_output_dict['additional_outputs'][output]['derivatives'][dv] = deriv

            # wrt additional_inputs
            for dv in self.additional_inputs:
                deriv = self.derivatives[( output , dv )]
                if hasattr(deriv,'tolist'):
                    deriv = deriv.tolist()
                remote_output_dict['additional_outputs'][output]['derivatives'][dv] = deriv

        return remote_output_dict

    def _gather_additional_input_derivatives_from_om_problem(self, remote_output_dict):
        responses = self.prob.model._responses
        for r in responses.keys():

            if responses[r]['type']=='obj':
                response_type = 'objective'
            elif responses[r]['type']=='con':
                response_type = 'constraints'

            for dv in self.additional_inputs:
                deriv = self.derivatives[( responses[r]['source'] , dv )]
                if hasattr(deriv,'tolist'):
                    deriv = deriv.tolist()
                remote_output_dict[response_type][r]['derivatives'][dv] = deriv
        return remote_output_dict

    def _gather_inputs_and_outputs_from_om_problem(self):
        remote_output_dict = self._gather_design_inputs_from_om_problem()
        remote_output_dict = self._gather_design_outputs_from_om_problem(remote_output_dict)
        remote_output_dict = self._gather_additional_inputs_from_om_problem(remote_output_dict)
        remote_output_dict = self._gather_additional_outputs_from_om_problem(remote_output_dict)
        if self.derivatives is not None:
            remote_output_dict = self._gather_design_derivatives_from_om_problem(remote_output_dict)
            remote_output_dict = self._gather_additional_output_derivatives_from_om_problem(remote_output_dict)
            remote_output_dict = self._gather_additional_input_derivatives_from_om_problem(remote_output_dict)
        remote_output_dict['design_counter'] = self.design_counter
        return remote_output_dict

    def _set_design_variables_into_the_server_problem(self, input_dict):
        design_changed = False
        for key in input_dict['design_vars'].keys():
            if (self.prob.get_val(key)!=input_dict['design_vars'][key]['val']).any():
                design_changed = True
            self.prob.set_val(key, input_dict['design_vars'][key]['val'])
        return design_changed

    def _set_additional_inputs_into_the_server_problem(self, input_dict, design_changed):
        for key in input_dict['additional_inputs'].keys():
            if (self.prob.get_val(key)!=input_dict['additional_inputs'][key]['val']).any():
                design_changed = True
            self.prob.set_val(key, input_dict['additional_inputs'][key]['val'])
        return design_changed

    def _save_additional_variable_names(self, input_dict):
        self.additional_inputs = input_dict['additional_inputs']
        self.additional_outputs = input_dict['additional_outputs']
        if hasattr(self.additional_inputs,'keys'):
            self.additional_inputs = list(self.additional_inputs.keys())

    def run(self):
        """
        Run the server.
        """
        while True:

            if self.rank==0:
                print('SERVER: Waiting for new design...', flush=True)

            command, input_dict = self._parse_incoming_message()

            # interpret command (options are "shutdown", "initialize", "evaluate", or "evaluate derivatives")
            if command=='shutdown':
                if self.rank==0:
                    print('SERVER: Received signal to shutdown', flush=True)
                break

            self._save_additional_variable_names(input_dict)

            if command=='initialize': # evaluate baseline model for RemoteComp setup
                if self.rank==0:
                    print('SERVER: Initialization requested... using baseline design', flush=True)
                if self.current_design_has_been_evaluated:
                    if self.rank==0:
                        print('SERVER: Design already evaluated, skipping run_model', flush=True)
                else:
                    self._run_model()
            else:
                design_changed = self._set_design_variables_into_the_server_problem(input_dict)
                design_changed = self._set_additional_inputs_into_the_server_problem(input_dict, design_changed)
                if design_changed:
                    self.current_design_has_been_evaluated = False
                    self.current_derivatives_have_been_evaluated = False

            if command=='evaluate derivatives': # compute derivatives
                if self.current_derivatives_have_been_evaluated:
                    if self.rank==0:
                        print('SERVER: Derivatives already evaluated, skipping compute_totals', flush=True)
                else:
                    if not self.current_design_has_been_evaluated:
                        if self.rank==0:
                            print('SERVER: Derivative needed, but design has changed... evaluating forward solution first', flush=True)
                        self._run_model()
                    if self.rank==0:
                        print('SERVER: Evaluating derivatives', flush=True)
                    self._compute_totals()

            elif command=='evaluate': # run model
                if self.current_design_has_been_evaluated:
                    if self.rank==0:
                        print('SERVER: Design already evaluated, skipping run_model', flush=True)
                else:
                    if self.rank==0:
                        print('SERVER: Evaluating design', flush=True)
                    self._run_model()

            # gather/return outputs
            output_dict = self._gather_inputs_and_outputs_from_om_problem()
            self._send_outputs_to_client(output_dict)

            # write current n2 with values
            om.n2(self.prob, show_browser=False, outfile=f"n2_inner_analysis_{input_dict['component_name']}.html")
