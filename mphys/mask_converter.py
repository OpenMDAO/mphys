import openmdao.api as om


class MaskedVariableDescription:
    """
    Attributes of a variable for conversion from serial to distributed
    or distributed to serial
    """

    def __init__(self, name: str, shape: tuple, tags=None):
        self.name = name
        self.shape = shape
        self.tags = tags


class MaskedConverter(om.ExplicitComponent):
    """

    """

    def initialize(self):
        self.options.declare(
            'input',
            desc='MaskedVariableDescription object of input that will be unmasked')
        self.options.declare(
            'output',
            desc='MaskedVariableDescription object of unmasked output')
        self.options.declare(
            'distributed', default=False,
            desc='Flag to determine if the inputs and outputs should be distributed arrays')
        self.options.declare(
            'mask',
            desc='masking array to apply to vectors. Contains boolean flags '
                 'indicating which indices should be included (True) or masked (False)')

    def setup(self):
        distributed = self.options['distributed']
        input = self.options['input']
        output = self.options['output']
        mask = self.options['mask']

        self.add_input(input.name, shape=input.shape, tags=input.tags, distributed=distributed)
        self.add_output(output.name, shape=output.shape, tags=output.tags, distributed=distributed)

    def compute(self, inputs, outputs):
        input = self.options['input']
        output = self.options['output']
        mask = self.options['mask']
        outputs[output.name] = inputs[input.name][mask]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        input = self.options['input']
        output = self.options['output']
        mask = self.options['mask']
        if mode == 'fwd':
            if input.name in d_inputs and output.name in d_outputs:
                d_outputs[output.name] += d_inputs[input.name][mask]

        if mode == 'rev':
            if input.name in d_inputs and output.name in d_outputs:
                d_inputs[input.name][mask] += d_outputs[output.name]

class UnmaskedConverter(om.ExplicitComponent):
    """

    """

    def initialize(self):
        self.options.declare(
            'input',
            desc='MaskedVariableDescription object of input that will be unmasked')
        self.options.declare(
            'output',
            desc='MaskedVariableDescription object of unmasked output')
        self.options.declare(
            'default_values', default= 0.0,
            desc='MaskedVariableDescription object of unmasked output')
        self.options.declare(
            'distributed', default=False,
            desc='Flag to determine if the inputs and outputs should be distributed arrays')
        self.options.declare(
            'mask',
            desc='masking array to apply to vectors. Contains boolean flags '
                 'indicating which indices should be included (True) or masked (False)')

    def setup(self):
        distributed = self.options['distributed']
        input = self.options['input']
        output = self.options['output']
        mask = self.options['mask']

        self.add_input(input.name, shape=input.shape, tags=input.tags, distributed=distributed)
        self.add_output(output.name, shape=output.shape, tags=output.tags, distributed=distributed)

    def compute(self, inputs, outputs):
        input = self.options['input']
        output = self.options['output']
        mask = self.options['mask']
        def_vals = self.options['default_values']
        outputs[output.name][:] = def_vals
        outputs[output.name][mask] = inputs[input.name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        input = self.options['input']
        output = self.options['output']
        mask = self.options['mask']
        if mode == 'fwd':
            if input.name in d_inputs and output.name in d_outputs:
                d_outputs[output.name][mask] += d_inputs[input.name]

        if mode == 'rev':
            if input.name in d_inputs and output.name in d_outputs:
                d_inputs[input.name] += d_outputs[output.name][mask]
