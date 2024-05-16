import numpy as np
import openmdao.api as om
from mphys.core.distributed_converter import DistributedVariableDescription


class DistributedSummer(om.ExplicitComponent):
    """
    An ExplicitComponent used to sum multiple distributed vectors together.
    This is a simple sum that results in a single distribute vector output (i.e. not a Reduce).
    """

    def initialize(self):
        self.options.declare("inputs", desc="List of DistibutedVariableDescription objects of inputs that will be summed", types=list)
        self.options.declare("output", desc="DistibutedVariableDescription object of summed output", types=DistributedVariableDescription)

    def setup(self):
        inputs = self.options["inputs"]
        output = self.options["output"]

        shape = inputs[0].shape
        for input in inputs:
            self.add_input(input.name, shape=input.shape, tags=input.tags, distributed=True)
            if input.shape != shape:
                raise ValueError("All input vectors must have the same shape on a processor")
        if output.shape != shape:
            raise ValueError("Output vectors must have the same shape as input vectors on a processor")
        self.add_output(output.name, shape=output.shape, tags=output.tags, distributed=True)

        self.inputs_names = [input_desc.name for input_desc in inputs]
        self.output_name = output.name

    def compute(self, inputs, outputs):
        outputs[self.output_name][:] = 0.0
        for input_name in self.inputs_names:
            outputs[self.output_name] += inputs[input_name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for input_name in self.inputs_names:
                if input_name in d_inputs and self.output_name in d_outputs:
                    d_outputs[self.output_name] += d_inputs[input_name]

        if mode == "rev":
            for input_name in self.inputs_names:
                if input_name in d_inputs and self.output_name in d_outputs:
                    d_inputs[input_name] += d_outputs[self.output_name]
