import numpy as np
import openmdao.api as om

class SumLoads(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('load_size', types=int)
        self.options.declare('load_list')

    def setup(self):

        for name in self.options['load_list']:
            self.add_input(name, shape_by_conn=True)

        self.add_output('F_summed',np.zeros(self.options['load_size']))

    def compute(self,inputs,outputs):

        outputs['F_summed'] = np.zeros(self.options['load_size'])
        for name in self.options['load_list']:
            outputs['F_summed'] += inputs[name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == 'fwd':
            pass

        if mode == 'rev':

            if 'F_summed' in d_outputs:
                for name in self.options['load_list']:
                    if name in d_inputs:
                        d_inputs[name] += d_outputs['F_summed']
