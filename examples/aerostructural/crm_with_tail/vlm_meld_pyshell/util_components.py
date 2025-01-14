import numpy as np
import openmdao.api as om
from mpi4py import MPI

from mphys import Builder

## lump DVs into a single scaling variable

class LumpDvs(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('N_dv', types=int)
        self.options.declare('dv', types=np.ndarray)

    def setup(self):
        self.add_input('lumped_dv',1.)
        self.add_output('dv_struct',np.zeros(self.options['N_dv']))

    def compute(self, inputs, outputs):
        outputs['dv_struct'] = inputs['lumped_dv']*self.options['dv']

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            if 'dv_struct' in d_outputs:
                if 'lumped_dv' in d_inputs:
                    d_inputs['lumped_dv'] += np.inner(d_outputs['dv_struct'],self.options['dv'])

# component which morphs the wing and/or tail sweep and span

class WingGeometry(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('name')
        self.options.declare('nnodes')
        self.options.declare('span')
        self.options.declare('coordinate_indices')

    def setup(self):
        self.set_check_partial_options(wrt='*',directional=True)

        self.input_name = 'x_'+self.options['name']+'_in'
        self.output_name = 'x_'+self.options['name']+'0'
        self.span = self.options['span']
        self.coordinate_indices = self.options['coordinate_indices']

        self.add_input('sweep_dv', val=[0.0]*len(self.coordinate_indices))
        self.add_input('span_dv', val=[0.0]*len(self.coordinate_indices))
        self.add_input(self.input_name, distributed=True, shape_by_conn=True)
        self.add_output(self.output_name,shape=self.options['nnodes']*3, distributed=True, tags=['mphys_coordinates'])

    def compute(self,inputs,outputs):

        for i in range(len(self.coordinate_indices)):

            x_0 = inputs[self.input_name][self.coordinate_indices[i]]

            if len(x_0) > 0:
                X = x_0[0::3]
                Y = x_0[1::3]
                Z = x_0[2::3]

                X = X + (Y/self.span)*inputs['sweep_dv'][i]
                Y = Y + (Y/self.span)*inputs['span_dv'][i]

                outputs[self.output_name][self.coordinate_indices[i][0::3]] = X
                outputs[self.output_name][self.coordinate_indices[i][1::3]] = Y
                outputs[self.output_name][self.coordinate_indices[i][2::3]] = Z

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':

            for i in range(len(self.coordinate_indices)):

                x_0 = inputs[self.input_name][self.coordinate_indices[i]]
                if len(x_0) > 0:
                    X = x_0[0::3]
                    Y = x_0[1::3]
                    Z = x_0[2::3]

                if self.output_name in d_outputs:
                    if 'sweep_dv' in d_inputs:
                        if len(x_0) > 0:
                            d_inputs['sweep_dv'][i] += np.inner(np.c_[Y/self.span,Y*0,Z*0].flatten(), d_outputs[self.output_name][self.coordinate_indices[i]])
                        d_inputs['sweep_dv'][i] = self.comm.allreduce(d_inputs['sweep_dv'][i], op=MPI.SUM)

                    if 'span_dv' in d_inputs:
                        if len(x_0) > 0:
                            d_inputs['span_dv'][i] += np.inner(np.c_[X*0,Y/self.span,Z*0].flatten(), d_outputs[self.output_name][self.coordinate_indices[i]])
                        d_inputs['span_dv'][i] = self.comm.allreduce(d_inputs['span_dv'][i], op=MPI.SUM)

# Builder
class GeometryBuilder(Builder):
    def __init__(self, builders, body_tags=[], span=29.39):
        self.builders = builders
        self.names = []
        self.nnodes = []
        self.body_tags = body_tags
        self.span = span

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        if len(self.nnodes)==0:
            coordinate_indices = {}
            for name, builder in self.builders.items():
                self.nnodes += [builder.get_number_of_nodes()]
                self.names += [name]

                # get coordinate indices for each body
                coordinate_indices[name] = []
                for body in self.body_tags:
                    grid_ids = builder.get_tagged_indices(body[name])
                    x_indices = np.sort(np.hstack([grid_ids*3, grid_ids*3+1, grid_ids*3+2])).tolist()
                    coordinate_indices[name] += [x_indices]

        geometry_morphing = om.Group()
        for i in range(len(self.nnodes)):
            geometry_morphing.add_subsystem(f'{self.names[i]}_geometry_morph',
                                            WingGeometry(name=self.names[i],
                                                         nnodes=self.nnodes[i],
                                                         span=self.span,
                                                         coordinate_indices=coordinate_indices[self.names[i]]),
                                            promotes=['*'])
        return geometry_morphing
