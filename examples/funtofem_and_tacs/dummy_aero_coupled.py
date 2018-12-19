from __future__ import division, print_function
import numpy as np

from openmdao.api import ImplicitComponent, ExplicitComponent
from tacs import TACS,functions

class AeroMesh(ExplicitComponent):
    """
    Dummy aerodynamic mesh reader

    Puts an aero node near the upper and lower surface of tip of the CRM wingbox
    """
    def initialize(self):
        self.options.declare('aero_mesh_setup', default = None, desc='Dummy call back to get aero object')

    def setup(self):
        self.flow = self.options['aero_mesh_setup'](self.comm)
        aero_nnodes = self.flow['nnodes']
        self.add_output('x_a0',shape=3*aero_nnodes, desc='aerodynamic surface coordinates')

    def compute(self,inputs,outputs):
        outputs['x_a0'] = np.array([46.4,29.2,4.5,46.4,29.2,4.75])

class AeroDeformer(ImplicitComponent):
    """
    Dummy aerodynamic deformer

    R = x_g - x_s - c1 = 0
    """
    def initialize(self):
        self.options.declare('aero_deformer_setup', default = None, desc='Dummy call back to get aero info')

        self.c1 = 1e-5

    def setup(self):
        flow = self.options['aero_deformer_setup'](self.comm)
        nnodes = flow['nnodes']

        self.add_input('x_a0',shape=nnodes*3, desc='dummy aero jig shape surface coordinates')
        self.add_input('x_a',shape=nnodes*3, desc='dummy aero deformed surface coordinates')
        self.add_output('x_g',shape=nnodes*3, desc='dummy aero volume grid')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x_g'] = outputs['x_g'] - inputs['x_a'] - self.c1

    def solve_nonlinear(self, inputs, outputs):
        outputs['x_g'] = inputs['x_a'] + self.c1

    def solve_linear(self, d_outputs,d_residuals,mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            d_residuals['x_g'] += d_outputs['x_g']

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if 'x_g' in d_residuals:
                if 'x_g' in d_outputs:
                    d_outputs['x_g'] += d_residuals['x_g']
                if 'x_a' in d_inputs:
                    d_inputs['x_a'] -= d_residuals['x_g']


class AeroSolver(ImplicitComponent):
    """
    Dummy aerodynamic solver

    R = q - c2 x_g = 0
    """
    def initialize(self):
        self.options.declare('aero_solver_setup', default = None, desc='Dummy call back to get aero info')

        self.c2 = 1e-5

    def setup(self):
        flow = self.options['aero_solver_setup'](self.comm)
        nnodes = flow['nnodes']

        self.add_input('dv_aero',shape=1, desc='dummy aero design variables')
        self.add_input('x_g',shape=nnodes*3, desc='dummy aero mesh')
        self.add_output('q',shape=nnodes*3, desc='dummy aero state')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['q'] = outputs['q'] - self.c2 * inputs['x_g']

    def solve_nonlinear(self, inputs, outputs):
        outputs['q'] = self.c2 * inputs['x_g']

    def solve_linear(self, d_outputs,d_residuals,mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            d_residuals['q'] += d_outputs['q']

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if 'q' in d_residuals:
                if 'q' in d_outputs:
                    d_outputs['q'] += d_residuals['q']
                if 'x_g' in d_inputs:
                    d_inputs['x_g'] -= self.c2 * d_residuals['q']

class AeroForceIntegrator(ExplicitComponent):
    """
    Dummy aerodynamic force integrator

    f_a = c3 q + c4 x_g
    """
    def initialize(self):
        self.options.declare('aero_force_integrator_setup', default = None, desc='Dummy call back to get aero info')

        self.c3 = 1e-5
        self.c4 = 1e-5

    def setup(self):
        flow = self.options['aero_force_integrator_setup'](self.comm)
        nnodes = flow['nnodes']

        self.add_input('x_g',shape=nnodes*3, desc='dummy aero mesh')
        self.add_input('q',shape=nnodes*3, desc='dummy aero state')
        self.add_output('f_a',shape=nnodes*3, desc='dummy aero force')

    def compute(self, inputs, outputs):
        outputs['f_a'] = self.c3 * inputs['q'] + self.c4 * inputs['x_g']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if 'f_a' in d_outputs:
                if 'q' in d_inputs:
                    d_inputs['q']   += self.c3 * d_outputs['f_a']
                if 'x_g' in d_inputs:
                    d_inputs['x_g'] += self.c4 * d_outputs['f_a']

