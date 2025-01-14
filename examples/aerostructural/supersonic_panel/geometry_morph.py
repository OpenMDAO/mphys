import numpy as np
import openmdao.api as om
from mpi4py import MPI

from mphys import Builder


# EC which morphs the geometry
class GeometryMorph(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('names')
        self.options.declare('n_nodes')

    def setup(self):
        self.add_input('geometry_morph_param')

        for name, n_nodes in zip(self.options['names'], self.options['n_nodes']):
            self.add_input(f'x_{name}_in', distributed=True, shape_by_conn=True)
            self.add_output(f'x_{name}0', shape=n_nodes*3, distributed=True, tags=['mphys_coordinates'])

    def compute(self,inputs,outputs):
        for name in self.options['names']:
            outputs[f'x_{name}0'] = inputs['geometry_morph_param']*inputs[f'x_{name}_in']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            for name in self.options['names']:
                if f'x_{name}0' in d_outputs:
                    if 'geometry_morph_param' in d_inputs:
                        d_inputs['geometry_morph_param'] += self.comm.allreduce(np.sum(d_outputs[f'x_{name}0']*inputs[f'x_{name}_in']), op=MPI.SUM)

                    if f'x_{name}_in' in d_inputs:
                        d_inputs[f'x_{name}_in'] += d_outputs[f'x_{name}0']*inputs['geometry_morph_param']


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
