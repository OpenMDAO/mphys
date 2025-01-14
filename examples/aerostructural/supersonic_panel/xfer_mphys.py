import numpy as np
import openmdao.api as om
from xfer import Xfer

from mphys import Builder


# EC which transfers displacements from structure to aero
class DispXfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.solver = self.options['solver']

        self.add_input('x_struct0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True, distributed=True, tags=['mphys_coupling'])
        self.add_output('u_aero', np.zeros(self.solver.aero.n_dof*self.solver.aero.n_nodes), distributed=True, tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        self.solver.xs = inputs['x_struct0']
        self.solver.xa = inputs['x_aero0']
        self.solver.us = inputs['u_struct']

        outputs['u_aero'] = self.solver.transfer_displacements() 
        
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            if 'u_aero' in d_outputs:
                d_xs, d_xa, d_us = self.solver.transfer_displacements_derivatives(
                    adjoint=d_outputs['u_aero']
                )

                if 'x_struct0' in d_inputs:
                    d_inputs['x_struct0'] += d_xs
                if 'x_aero0' in d_inputs:
                    d_inputs['x_aero0'] += d_xa 
                if 'u_struct' in d_inputs:
                    d_inputs['u_struct'] += d_us


# EC which transfers loads from aero to structure
class LoadXfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.solver = self.options['solver']

        self.add_input('x_struct0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('f_aero', shape_by_conn=True, distributed=True, tags=['mphys_coupling'])
        self.add_output('f_struct', np.zeros(self.solver.struct.n_dof*self.solver.struct.n_nodes), distributed=True, tags=['mphys_coupling'])
 
    def compute(self,inputs,outputs):
        self.solver.xs = inputs['x_struct0']
        self.solver.xa = inputs['x_aero0']
        self.solver.fa = inputs['f_aero']

        outputs['f_struct'] = self.solver.transfer_loads()
        
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':            
            if 'f_struct' in d_outputs:
                d_xs, d_xa, d_fa = self.solver.transfer_loads_derivatives(
                    adjoint=d_outputs['f_struct']
                )

                if 'x_struct0' in d_inputs:
                    d_inputs['x_struct0'] += d_xs
                if 'x_aero0' in d_inputs:
                    d_inputs['x_aero0'] += d_xa
                if 'f_aero' in d_inputs:
                    d_inputs['f_aero'] += d_fa


# Builder
class XferBuilder(Builder):
    def __init__(self, aero_builder, struct_builder):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder

    def initialize(self, comm): 
        self.solver = Xfer(
            aero = self.aero_builder.solver,
            struct = self.struct_builder.solver,
            comm = comm
        )

    def get_coupling_group_subsystem(self, scenario_name=None):
        disp_xfer = DispXfer(solver=self.solver)
        load_xfer = LoadXfer(solver=self.solver)
        return disp_xfer, load_xfer


