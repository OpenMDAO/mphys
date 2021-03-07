from __future__ import division, print_function
import numpy as np

import openmdao.api as om

from vlm_solver import VLM_solver, VLM_forces

class VlmMesh(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('N_nodes')
        self.options.declare('x_aero0')

    def setup(self):

        N_nodes = self.options['N_nodes']
        self.x_a0 = self.options['x_aero0']
        self.add_output('x_aero0',np.zeros(N_nodes*3), tags='solver')

    def mphys_add_coordinate_input(self):

        N_nodes = self.options['N_nodes']
        self.add_input('x_aero0_points',np.zeros(N_nodes*3))
        return 'x_aero0_points', self.x_a0

    def compute(self,inputs,outputs):

        if 'x_aero0_points' in inputs:
            outputs['x_aero0'] = inputs['x_aero0_points']
        else:
            outputs['x_aero0'] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'x_aero0_points' in d_inputs:
                d_outputs['x_aero0'] += d_inputs['x_aero0_points']
        elif mode == 'rev':
            if 'x_aero0_points' in d_inputs:
                d_inputs['x_aero0_points'] += d_outputs['x_aero0']

class GeoDisp(om.ExplicitComponent):
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

        self.add_input('x_aero0',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')
        self.add_input('u_aero', shape=local_size,val=np.zeros(local_size),src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface displacements')

        self.add_output('x_aero',shape=local_size,desc='deformed aerodynamic surface')

    def compute(self,inputs,outputs):
        outputs['x_aero'] = inputs['x_aero0'] + inputs['u_aero']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_aero' in d_outputs:
                if 'x_aero0' in d_inputs:
                    d_outputs['x_aero'] += d_inputs['x_aero0']
                if 'u_aero' in d_inputs:
                    d_outputs['x_aero'] += d_inputs['u_aero']
        if mode == 'rev':
            if 'x_aero' in d_outputs:
                if 'x_aero0' in d_inputs:
                    d_inputs['x_aero0'] += d_outputs['x_aero']
                if 'u_aero' in d_inputs:
                    d_inputs['u_aero']  += d_outputs['x_aero']

class VlmGroup(om.Group):

    def initialize(self):
        self.options.declare('options_dict')
        # Flag to enable AS coupling items
        # TODO we do not use this now and assume we have as_coupling = True always.
        # this needs to be updated to run aero-only vlm
        self.options.declare('as_coupling')

    def setup(self):

        options_dict = self.options['options_dict']
        # this can be done much more cleanly with **options_dict
        N_nodes    = options_dict['N_nodes']
        N_elements = options_dict['N_elements']
        x_a0       = options_dict['x_aero0']
        quad       = options_dict['quad']

        # by default, we use nodal forces. however, if the user wants to use
        # tractions, they can specify it in the options_dict
        compute_traction = False
        if 'compute_traction' in options_dict:
            compute_traction = options_dict['compute_traction']

        self.add_subsystem('geo_disp', GeoDisp(nnodes=N_nodes), promotes_inputs=['u_aero', 'x_aero0'])

        self.add_subsystem('solver', VLM_solver(
            N_nodes=N_nodes,
            N_elements=N_elements,
            quad=quad),
            promotes_inputs=['aoa','mach'])

        self.add_subsystem('forces', VLM_forces(
            N_nodes=N_nodes,
            N_elements=N_elements,
            quad=quad,
            compute_traction=compute_traction),
            promotes_inputs=['mach','q_inf','vel','mu'],
            promotes_outputs=[('fa','f_aero'),'CL','CD'])

    def configure(self):
        self.connect('geo_disp.x_aero', ['solver.xa', 'forces.xa'])
        self.connect('solver.Cp', 'forces.Cp')

class DummyVlmSolver(object):
    '''
    a dummy object that can be used to hold the
    memory associated with a single VLM solver so that
    multiple OpenMDAO components share the same memory.
    '''
    def __init__(self, options, comm):
        self.options = options
        self.comm = comm
        self.allWallsGroup = 'allWalls'

    # the methods below here are required for RLT
    def getSurfaceCoordinates(self, group):
        # just return the full coordinates
        return self.options['x_aero0']

    def getSurfaceConnectivity(self, group):
        # -1 for the conversion between fortran and C
        conn = self.options['quad'].copy() -1
        faceSizes = 4*np.ones(len(conn), 'intc')
        return conn.astype('intc'), faceSizes

class VlmBuilder(object):

    def __init__(self, options):
        self.options = options

    def init_solver(self, comm):
        self.solver = DummyVlmSolver(options=self.options, comm=comm)

    def get_solver(self):
        return self.solver

    def get_element(self, **kwargs):
        return VlmGroup(options_dict=self.options, **kwargs)

    def get_mesh_element(self):
        N_nodes = self.options['N_nodes']
        x_aero0 = self.options['x_aero0']
        return VlmMesh(N_nodes=N_nodes, x_aero0=x_aero0)

    def get_nnodes(self):
        return self.options['N_nodes']
