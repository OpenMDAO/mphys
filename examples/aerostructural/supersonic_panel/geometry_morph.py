import numpy as np
import openmdao.api as om
from mpi4py import MPI
from mphys import Builder, MPhysVariables


X_AERO0_GEOM_INPUT = MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT
X_AERO0_GEOM_OUTPUT = MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT

X_STRUCT_GEOM_INPUT = MPhysVariables.Structures.Geometry.COORDINATES_INPUT
X_STRUCT_GEOM_OUTPUT = MPhysVariables.Structures.Geometry.COORDINATES_OUTPUT

# EC which morphs the geometry
class GeometryMorph(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('names')
        self.options.declare('n_nodes')

        self.input_names = {'aero': X_AERO0_GEOM_INPUT, 'struct': X_STRUCT_GEOM_INPUT}
        self.output_names = {'aero': X_AERO0_GEOM_OUTPUT, 'struct': X_STRUCT_GEOM_OUTPUT}

    def setup(self):
        self.add_input('geometry_morph_param')

        for name, n_nodes in zip(self.options['names'], self.options['n_nodes']):
            self.add_input(self.input_names[name], distributed=True, shape_by_conn=True, tags=['mphys_coordinates'])
            self.add_output(self.output_names[name], shape=n_nodes*3, distributed=True, tags=['mphys_coordinates'])

    def compute(self,inputs,outputs):
        for name in self.options['names']:
            outputs[self.output_names[name]] = inputs['geometry_morph_param']*inputs[self.input_names[name]]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            for name in self.options['names']:
                if self.output_names[name] in d_outputs:
                    if 'geometry_morph_param' in d_inputs:
                        d_inputs['geometry_morph_param'] += self.comm.allreduce(np.sum(d_outputs[self.output_names[name]]*inputs[self.input_names[name]]), op=MPI.SUM)

                    if self.input_names[name] in d_inputs:
                        d_inputs[self.input_names[name]] += d_outputs[self.output_names[name]]*inputs['geometry_morph_param']


# Builder
class GeometryBuilder(Builder):
    def __init__(self, builders):
        self.builders = builders
        self.names = None

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        if self.names is None:
            self.names = []
            self.n_nodes = []
            for name, builder in self.builders.items():
                self.names.append(name)
                self.n_nodes.append(builder.get_number_of_nodes())
        return GeometryMorph(names=self.names, n_nodes=self.n_nodes)
