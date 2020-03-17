from __future__ import division, print_function
import numpy as np

import openmdao.api as om
from openmdao.api import ImplicitComponent, ExplicitComponent, Group
from omfsi.geo_disp import Geo_Disp
from vlm_solver import VLM_solver, VLM_forces

class VlmMesh(ExplicitComponent):

    def initialize(self):
        self.options.declare('N_nodes')
        self.options.declare('x_a0')

    def setup(self):

        N_nodes = self.options['N_nodes']
        self.x_a0 = self.options['x_a0']
        self.add_output('x_a0_mesh',np.zeros(N_nodes*3))

    def compute(self,inputs,outputs):

        outputs['x_a0_mesh'] = self.x_a0


class VLM_group(Group):

    def initialize(self):
        self.options.declare('options_dict')

    def setup(self):

        options_dict = self.options['options_dict']
        # this can be done much more cleanly with **options_dict
        mach       = options_dict['mach']
        alpha      = options_dict['alpha']
        q_inf      = options_dict['q_inf']
        vel        = options_dict['vel']
        mu         = options_dict['mu']
        N_nodes    = options_dict['N_nodes']
        N_elements = options_dict['N_elements']
        x_a0       = options_dict['x_a0']
        quad       = options_dict['quad']

        self.add_subsystem('geo_disp', Geo_Disp(nnodes=N_nodes), promotes_inputs=['u_a', 'x_a0'])

        self.add_subsystem('solver', VLM_solver(
            N_nodes=N_nodes,
            N_elements=N_elements,
            quad=quad,
            mach=mach
        ), promotes_inputs=['alpha'])

        self.add_subsystem('funcs', VLM_forces(
            N_nodes=N_nodes,
            N_elements=N_elements,
            quad=quad,
            q_inf=q_inf,
            mach=mach,
            vel=vel,
            mu=mu
        ), promotes_outputs=[('fa','f_a')])

    def configure(self):
        self.connect('geo_disp.x_a', ['solver.xa', 'funcs.xa'])
        self.connect('solver.Cp', 'funcs.Cp')

class dummyVLMSolver(object):
    '''
    a dummy object that can be used to hold the
    memory associated with a single VLM solver so that
    multiple OpenMDAO components share the same memory.
    '''
    def __init__(self, options, comm):
        self.options = options
        self.comm = comm

class VLM_builder(object):

    def __init__(self, options):
        self.options = options

    # api level method for all builders
    def init_solver(self, comm):
        self.solver = dummyVLMSolver(options=self.options, comm=comm)

    # api level method for all builders
    def get_solver(self):
        return self.solver

    # api level method for all builders
    def get_element(self):
        return VLM_group(options_dict=self.options)

    def get_mesh_element(self):
        N_nodes = self.options['N_nodes']
        x_a0 = self.options['x_a0']
        return VlmMesh(N_nodes=N_nodes, x_a0=x_a0)

    def get_nnodes(self):
        return self.options['N_nodes']