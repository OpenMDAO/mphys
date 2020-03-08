from __future__ import division, print_function
import numpy as np

from openmdao.api import ImplicitComponent, ExplicitComponent, Group
from omfsi.assembler import OmfsiSolverAssembler, OmfsiAssembler
"""
Fake FSI components for testing, debugging etc
"""

class FakeStructAssembler(OmfsiSolverAssembler):
    def __init__(self,c0,c1,symmetry,nodes=None):
        self.c0 = c0
        self.c1 = c1
        self.symmetry = symmetry

        self.nodes = nodes
        if self.nodes is None:
            self.nodes = np.array([0.0,0.0,0.0,
                                   0.0,1.0,0.0,
                                   1.0,1.0,0.0,
                                   1.0,0.0,0.0])
        self.nnodes = int(self.nodes.size / 3)
        self.ndof = 3

    def get_nnodes(self):
        return self.nnodes

    def get_ndof(self):
        return self.ndof

    def add_model_components(self,model,connection_srcs):
        pass

    def add_scenario_components(self,model,scenario,connection_srcs):
        pass

    def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):
        fake_struct = FakeStructSolver(self.nnodes, self.c0, self.c1, self.symmetry)
        fsi_group.add_subsystem('fake_struct',fake_struct)

        connection_srcs['u_s'] = scenario.name+'.'+fsi_group.name+'.fake_struct.u_s'

    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        model.connect(connection_srcs['f_s'],scenario.name+'.'+fsi_group.name+'.fake_struct.f_s')

class FakeStructSolver(ExplicitComponent):
    """
    Fake structural model u_s = c0 + c1 * f_s
    """
    def initialize(self):
        self.options.declare('nnodes',default=None, desc='number of struct nodes')
        self.options.declare('c0', default= 0.0, desc='constant offset')
        self.options.declare('c1', default= 0.0, desc='linear coeff')
        self.options.declare('symmetry', default = False, desc='symmetry-only use z equation')

    def setup(self):
        self.add_input('f_s',shape=int(3*self.options['nnodes']))
        self.add_output('u_s',shape=int(3*self.options['nnodes']))

    def _symmetry_indices(self):
        if self.options['symmetry']:
            start = 2
            skip  = 3
        else:
            start = 0
            skip  = 1
        return start, skip

    def compute(self,inputs,outputs):
        start, skip = self._symmetry_indices()
        c0 = self.options['c0']
        c1 = self.options['c1']

        outputs['u_s'][start::skip] = c0 + c1 * inputs['f_s'][start::skip]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs, mode):
        start, skip = self._symmetry_indices()
        c0 = self.options['c0']
        c1 = self.options['c1']

        if mode == 'fwd':
            if 'u_s' in d_outputs:
                if 'f_s' in d_inputs:
                    d_outputs['u_s'][start::skip] = c1 * d_inputs['f_s'][start::skip]
        if mode == 'rev':
            if 'u_s' in d_outputs:
                if 'f_s' in d_inputs:
                    d_inputs['f_s'][start::skip] = c1 * d_outputs['u_s'][start::skip]

class FakeAeroAssembler(OmfsiSolverAssembler):
    def __init__(self,c0,c1,symmetry,nodes=None):
        self.c0 = c0
        self.c1 = c1
        self.symmetry = symmetry

        self.nodes = nodes
        if self.nodes is None:
            # offset from z=0 to force the load and displacement transfer to do something
            self.nodes = np.array([0.0,0.0,1e-6,
                                   0.0,1.0,1e-6,
                                   0.5,0.5,1e-6,
                                   1.0,1.0,1e-6,
                                   1.0,0.0,1e-6])
        self.nnodes = int(self.nodes.size / 3)

    def get_nnodes(self):
        return self.nnodes

    def add_model_components(self,model,connection_srcs):
        pass

    def add_scenario_components(self,model,scenario,connection_srcs):
        pass

    def add_fsi_components(self,model,scenario,fsi_groupconnection_srcs):
        fake_aero = FakeAeroSolver(self.nnodes, self.c0, self.c1, self.symmetry)
        fsi_group.add_subsystem('fake_aero',fake_aero)

        connection_srcs['f_a'] = scenario.name+'.'+fsi_group.name+'.fake_aero.f_a'

    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        model.connect(connection_srcs['x_a'],scenario.name+'.'+fsi_group.name+'.fake_aero.x_a')

class FakeAeroSolver(ExplicitComponent):
    """
    Fake aero model f_a = c0 + c1 * x_a
    """
    def initialize(self):
        self.options.declare('nnodes',default=None, desc='number of aero nodes')
        self.options.declare('c0', default= 0.0, desc='constant offset')
        self.options.declare('c1', default= 0.0, desc='linear coeff')
        self.options.declare('symmetry', default = False, desc='symmetry-only use z equation')

    def setup(self):
        self.add_input('x_a',shape=int(3*self.options['nnodes']))
        self.add_output('f_a',shape=int(3*self.options['nnodes']))

    def _symmetry_indices(self):
        if self.options['symmetry']:
            start = 2
            skip  = 3
        else:
            start = 0
            skip  = 1
        return start, skip

    def compute(self,inputs,outputs):
        start, skip = self._symmetry_indices()
        c0 = self.options['c0']
        c1 = self.options['c1']

        outputs['f_a'][start::skip] = c0 + c1 * inputs['x_a'][start::skip]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs, mode):
        start, skip = self._symmetry_indices()
        c0 = self.options['c0']
        c1 = self.options['c1']

        if mode == 'fwd':
            if 'f_a' in d_outputs:
                if 'x_a' in d_inputs:
                    d_outputs['f_a'][start::skip] = c1 * d_inputs['x_a'][start::skip]
        if mode == 'rev':
            if 'f_a' in d_outputs:
                if 'x_a' in d_inputs:
                    d_inputs['x_a'][start::skip] = c1 * d_outputs['f_a'][start::skip]

class FakeXferAssembler(OmfsiAssembler):
    def __init__(self,struct_assembler,aero_assembler):
        self.struct_ndof   = self.struct_assembler.get_ndof()
        self.struct_nnodes = self.struct_assembler.get_nnodes()
        self.aero_nnodes   = self.aero_assembler.get_nnodes()

    def set_disp_xfer_properties(c0=0.0,c1=0.0,symmetry=False):
        self.c0_disp = c0
        self.c1_disp = c1
        self.symmtery_disp = symmtery

    def set_load_xfer_properties(c0=0.0,c1=0.0,symmetry=False):
        self.c0_load = c0
        self.c1_load = c1
        self.symmtery_load = symmtery

    def add_model_components(self,model,connection_srcs):
        pass

    def add_scenario_components(self,model,scenario,connection_srcs):
        pass

    def add_fsi_components(self,model,scenario,fsi_groupconnection_srcs):
        fake_disp_xfer = FakeDispXfer(self.c0_disp, self.c1_disp, self.symmetry_disp,
                                      self.struct_ndof, self.struct_nnodes, self.aero_nnodes)
        fake_load_xfer = FakeLoadXfer(self.c0_load, self.c1_load, self.symmetry_load,
                                      self.struct_ndof, self.struct_nnodes, self.aero_nnodes)
        fsi_group.add_subsystem('fake_disp_xfer',fake_disp_xfer)
        fsi_group.add_subsystem('fake_load_xfer',fake_load_xfer)

        connection_srcs['u_a'] = scenario.name+'.'+fsi_group.name+'.fake_disp_xfer.u_a'
        connection_srcs['f_s'] = scenario.name+'.'+fsi_group.name+'.fake_load_xfer.f_s'

    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        model.connect(connection_srcs['u_s'],scenario.name+'.'+fsi_group.name+'.fake_disp_xfer.u_s')
        model.connect(connection_srcs['f_a'],scenario.name+'.'+fsi_group.name+'.fake_aero.f_a')

class FakeDispXfer(ExplicitComponent):
    """
    Fake displacement transfer model u_a_{x,y,z} = c0 + c1 * Sum(u_s_{x,y,z})
    Sum because the vectors may be different sizes
    """
    def initialize(self):
        self.options.declare('struct_ndof',default=0, desc='number of struct dofs')
        self.options.declare('struct_nnodes',default=0, desc='number of struct nodes')
        self.options.declare('aero_nnodes',default=0, desc='number of aero nodes')
        self.options.declare('c0', default= 0.0, desc='constant offset')
        self.options.declare('c1', default= 0.0, desc='linear coeff')
        self.options.declare('symmetry', default = False, desc='symmetry-only use z equation')

    def setup(self):
        self.add_input('u_s',shape=int(self.options['struct_ndof']*self.options['struct_nnodes']))
        self.add_output('u_a',shape=int(3*self.options['aero_nnodes']))

        self.start_indices = [2] if self.options['symmetry'] else [0,1,2]

    def compute(self,inputs,outputs):
        c0 = self.options['c0']
        c1 = self.options['c1']

        for start in self.start_indices:
            outputs['u_a'][start::3] = c0 + c1 * np.sum(inputs['u_s'][start::3])

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs, mode):
        c0 = self.options['c0']
        c1 = self.options['c1']

        if mode == 'fwd':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    for start in self.start_indices:
                        d_outputs['u_a'][start::3] = c1 * np.sum(d_inputs['u_s'][start::3])
        if mode == 'rev':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    for start in self.start_indices:
                        d_inputs['u_s'][start::3] = c1 * np.sum(d_outputs['u_a'][start::3])

class FakeLoadXfer(ExplicitComponent):
    """
    Fake load transfer model f_s_{x,y,z} = c0 + c1 * Sum(f_a_{x,y,z})
    Sum because the vectors may be different sizes
    """
    def initialize(self):
        self.options.declare('struct_ndof',default=0, desc='number of struct dofs')
        self.options.declare('struct_nnodes',default=0, desc='number of struct nodes')
        self.options.declare('aero_nnodes',default=0, desc='number of aero nodes')
        self.options.declare('c0', default= 0.0, desc='constant offset')
        self.options.declare('c1', default= 0.0, desc='linear coeff')
        self.options.declare('symmetry', default = False, desc='symmetry-only use z equation')

    def setup(self):
        self.add_input('f_a',shape=int(3*self.options['aero_nnodes']))
        self.add_output('f_s',shape=int(self.options['struct_ndof']*self.options['struct_nnodes']))

        self.start_indices = [2] if self.options['symmetry'] else [0,1,2]

    def compute(self,inputs,outputs):
        c0 = self.options['c0']
        c1 = self.options['c1']

        for start in self.start_indices:
            outputs['f_s'][start::3] = c0 + c1 * np.sum(inputs['f_a'][start::3])

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs, mode):
        c0 = self.options['c0']
        c1 = self.options['c1']

        if mode == 'fwd':
            if 'f_s' in d_outputs:
                if 'f_a' in d_inputs:
                    for start in self.start_indices:
                        d_outputs['f_s'][start::3] = c1 * np.sum(d_inputs['f_a'][start::3])
        if mode == 'rev':
            if 'f_s' in d_outputs:
                if 'f_a' in d_inputs:
                    for start in self.start_indices:
                        d_inputs['f_a'][start::3] = c1 * np.sum(d_outputs['f_s'][start::3])
