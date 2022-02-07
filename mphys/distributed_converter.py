import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI


class DistributedVariableDescription:
    """
    Attributes of a variable for conversion from serial to distributed
    or distributed to serial
    """

    def __init__(self, name: str, shape: tuple, tags=[]):
        self.name = name
        self.shape = shape
        self.tags = tags


class DistributedConverter(om.ExplicitComponent):
    """
    An ExplicitComponent to convert from distributed to serial and serial to distributed variables.
    Mphys requires the coupling inputs and outputs to be distributed variables, so this
    class is provided to help with those conversions.
    For each mphys variable, a {variable}_serial version is created for the nonparallel solver to connect to and the
    distributed version will have the full vector on the root processor and zero length on the other processors.
    Given a list of distributed inputs in the options, the component will add variables to the inputs as distributed and
    produce {variable}_serial as outputs.
    Given a list of distributed outputs in the options, the component will add variables to the outputs as distributed and
    add {variable}_serial as inputs.

    """

    def initialize(self):
        self.options.declare(
            'distributed_inputs', default=[],
            desc='List of DistributedVariableDescription objects that will be converted from distributed to serial')
        self.options.declare(
            'distributed_outputs', default=[],
            desc='List of DistributedVariableDescription objects that will be converted from serial to distributed')

    def setup(self):
        for input in self.options['distributed_inputs']:
            self.add_input(input.name, shape_by_conn=True, tags=input.tags, distributed=True)
            self.add_output(f'{input.name}_serial', shape=input.shape, distributed=False)

        for output in self.options['distributed_outputs']:
            shape = output.shape if self.comm.Get_rank() == 0 else 0
            self.add_input(f'{output.name}_serial', shape_by_conn=True, distributed=False)
            self.add_output(output.name, shape=shape, tags=output.tags, distributed=True)

    def compute(self, inputs, outputs):
        for input in self.options['distributed_inputs']:
            if self.comm.Get_rank() == 0:
                outputs[f'{input.name}_serial'] = inputs[input.name]
            outputs[f'{input.name}_serial'] = self.comm.bcast(outputs[f'{input.name}_serial'])

        for output in self.options['distributed_outputs']:
            if self.comm.Get_rank() == 0:
                outputs[output.name] = inputs[f'{output.name}_serial']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            for input in self.options['distributed_inputs']:
                if input.name in d_inputs and f'{input.name}_serial' in d_outputs:
                    if self.comm.Get_rank() == 0:
                        d_outputs[f'{input.name}_serial'] += d_inputs[input.name]
                    d_outputs[f'{input.name}_serial'] = self.comm.bcast(
                        d_outputs[f'{input.name}_serial'])

            for output in self.options['distributed_outputs']:
                if output.name in d_outputs and f'{output.name}_serial' in d_inputs:
                    if self.comm.Get_rank() == 0:
                        d_outputs[output.name] += d_inputs[f'{output.name}_serial']

        if mode == 'rev':
            for input in self.options['distributed_inputs']:
                if input.name in d_inputs and f'{input.name}_serial' in d_outputs:
                    if MPI and self.comm.size > 1:
                        full = np.zeros(d_outputs[f'{input.name}_serial'].size)
                        self.comm.Reduce(d_outputs[f'{input.name}_serial'], full, op=MPI.SUM)
                        if self.comm.Get_rank() == 0:
                            d_inputs[input.name] += full
                    else:
                        d_inputs[input.name] += d_outputs[f'{input.name}_serial']

            for output in self.options['distributed_outputs']:
                if output.name in d_outputs and f'{output.name}_serial' in d_inputs:
                    if self.comm.Get_rank() == 0:
                        d_inputs[f'{output.name}_serial'] += d_outputs[output.name]
