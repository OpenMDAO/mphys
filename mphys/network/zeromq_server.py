import openmdao.api as om
import zmq, json, warnings

class MPhysZeroMQServer:
    def __init__(self, port, get_om_group_function_pointer,
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

        self._load_the_model()
        self._setup_zeromq_socket(port)

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

    def _setup_zeromq_socket(self, port):
        if self.rank==0:
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{port}")

    def _parse_incoming_message(self):
        inputs = None
        if self.rank==0:
            inputs = self.socket.recv().decode()
        inputs = self.prob.model.comm.bcast(inputs)

        command, input_dict = inputs.split("|")
        if 'evaluate' in command:
            input_dict = json.loads(input_dict)
        return command, input_dict

    def _set_design_variables_into_the_server_problem(self, input_dict):
        design_changed = False
        for key in input_dict['design_vars'].keys():
            if (self.prob.get_val(key)!=input_dict['design_vars'][key]['val']).any():
                design_changed = True
            self.prob.set_val(key, input_dict['design_vars'][key]['val'])
        return design_changed

    def _run_model(self):
        if self.ignore_runtime_warnings:
            with warnings.catch_warnings(record=True) as w:
                self.prob.run_model()
        else:
            self.prob.run_model()
        self.current_design_has_been_evaluated = True
        self.derivatives = None

    def _compute_totals(self):
        if self.ignore_runtime_warnings:
            with warnings.catch_warnings(record=True) as w:
                self.derivatives = self.prob.compute_totals()
        else:
            self.derivatives = self.prob.compute_totals()
        self.current_derivatives_have_been_evaluated = True

    def _gather_inputs_and_outputs_from_om_problem(self):

        # save inputs (TODO: may want to get adder, scaler, etc.)
        design_vars = self.prob.model._design_vars
        output_dict = {'design_vars':{}}
        for dv in design_vars.keys():
            output_dict['design_vars'][dv] = {'val': self.prob.get_val(dv),
                                              'ref': design_vars[dv]['ref'],
                                              'lower': design_vars[dv]['lower'],
                                              'upper': design_vars[dv]['upper'],
                                              'units': design_vars[dv]['units']}

            # apply reference value
            if output_dict['design_vars'][dv]['ref'] is None:
                output_dict['design_vars'][dv]['ref'] = 1.0
            output_dict['design_vars'][dv]['lower'] *= output_dict['design_vars'][dv]['ref']
            output_dict['design_vars'][dv]['upper'] *= output_dict['design_vars'][dv]['ref']

            # convert to lists for json input/output
            for key in output_dict['design_vars'][dv].keys():
                if hasattr(output_dict['design_vars'][dv][key], 'tolist'):
                    output_dict['design_vars'][dv][key] = output_dict['design_vars'][dv][key].tolist()

        # save outputs
        responses = self.prob.model._responses
        output_dict.update({'objective':{}, 'constraints':{}})
        for r in responses.keys():

            if responses[r]['type']=='obj':
                response_type = 'objective'
            elif responses[r]['type']=='con':
                response_type = 'constraints'

            output_dict[response_type][r] = {'val': self.prob.get_val(r, get_remote=True),
                                             'ref': responses[r]['ref']}

            if response_type=='constraints': # get constraint bounds
                output_dict[response_type][r].update({'lower': responses[r]['lower'],
                                                      'upper': responses[r]['upper'],
                                                      'equals': responses[r]['equals']})

                # apply reference value
                if output_dict[response_type][r]['ref'] is None:
                    output_dict[response_type][r]['ref'] = 1.0
                if output_dict[response_type][r]['equals'] is not None: # equality constraint
                    output_dict[response_type][r]['equals'] *= output_dict[response_type][r]['ref']
                else:
                    if output_dict[response_type][r]['lower']>-1e20:
                        output_dict[response_type][r]['lower'] *= output_dict[response_type][r]['ref']
                    if output_dict[response_type][r]['upper']<1e20:
                        output_dict[response_type][r]['upper'] *= output_dict[response_type][r]['ref']

            # convert to lists for json input/output
            for key in output_dict[response_type][r].keys():
                if hasattr(output_dict[response_type][r][key], 'tolist'):
                    output_dict[response_type][r][key] = output_dict[response_type][r][key].tolist()

            if self.derivatives is not None: # save derivatives
                output_dict[response_type][r]['derivatives'] = {}
                for dv in design_vars.keys():
                    deriv = self.derivatives[(responses[r]['source'], design_vars[dv]['source'])]
                    if hasattr(deriv,'tolist'):
                        deriv = deriv.tolist()
                    output_dict[response_type][r]['derivatives'][dv] = deriv

        return output_dict

    def _send_outputs_to_client(self, output_dict: dict):
        if self.rank==0:
            self.socket.send(str(json.dumps(output_dict)).encode())

    def run(self):

        # loop through and wait for new designs to evaluate
        while 1:

            if self.rank==0:
                print('SERVER: Waiting for new design...', flush=True)

            # get inputs from client
            command, input_dict = self._parse_incoming_message()

            # interpret command (options are "shutdown", "initialize", "evaluate", or "evaluate derivatives")
            if command=='shutdown':
                if self.rank==0:
                    print('SERVER: Received signal to shutdown', flush=True)
                break

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
            om.n2(self.prob, show_browser=False, outfile='n2_inner_analysis.html')
