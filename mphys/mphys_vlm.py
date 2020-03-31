from __future__ import division, print_function
import numpy as np

import openmdao.api as om

# from mphys.geo_disp import Geo_Disp
from vlm_solver import VLM_solver, VLM_forces

class VlmMesh(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('N_nodes')
        self.options.declare('x_a0')

    def setup(self):

        N_nodes = self.options['N_nodes']
        self.x_a0 = self.options['x_a0']
        self.add_output('x_a0_mesh',np.zeros(N_nodes*3))

    def compute(self,inputs,outputs):

        outputs['x_a0_mesh'] = self.x_a0

class Geo_Disp(om.ExplicitComponent):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options['distributed'] = True
        self.options.declare('nnodes')

    def setup(self):
        aero_nnodes = self.options['nnodes']
        local_size = aero_nnodes * 3
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a0',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')
        self.add_input('u_a', shape=local_size,val=np.zeros(local_size),src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface displacements')

        self.add_output('x_a',shape=local_size,desc='deformed aerodynamic surface')

    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_outputs['x_a'] += d_inputs['x_a0']
                if 'u_a' in d_inputs:
                    d_outputs['x_a'] += d_inputs['u_a']
        if mode == 'rev':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_inputs['x_a0'] += d_outputs['x_a']
                if 'u_a' in d_inputs:
                    d_inputs['u_a']  += d_outputs['x_a']

class VLM_group(om.Group):

    def initialize(self):
        self.options.declare('options_dict')
        # Flag to enable AS coupling items
        # TODO we do not use this now and assume we have as_coupling = True always.
        # this needs to be updated to run aero-only vlm
        self.options.declare('as_coupling')

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

    def init_solver(self, comm):
        self.solver = dummyVLMSolver(options=self.options, comm=comm)

    def get_solver(self):
        return self.solver

    def get_element(self, **kwargs):
        return VLM_group(options_dict=self.options, **kwargs)

    def get_mesh_element(self):
        N_nodes = self.options['N_nodes']
        x_a0 = self.options['x_a0']
        return VlmMesh(N_nodes=N_nodes, x_a0=x_a0)

    def get_nnodes(self):
        return self.options['N_nodes']
