#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import openmdao.api as om
from tacs import TACS, elements, functions
from omfsi.tacs_component_configure import TacsMass, TacsFunctions

class ModalStructAssembler(OmfsiSolverAssembler):
    def __init__(self,solver_options):
        self.add_elements = solver_options['add_elements']
        self.mesh_file    = solver_options['mesh_file']
        self.nmodes       = solver_options['nmodes']
        self.get_funcs    = solver_options['get_funcs']
        self.f5_writer    = solver_options['f5_writer']

        self.tacs = None

    def get_tacs(self,comm):
        if self.tacs is None:
            self.comm = comm
            mesh = TACS.MeshLoader(comm)
            mesh.scanBDFFile(self.mesh_file)

            self.ndof, self.ndv = self.add_elements(mesh)

            self.tacs = mesh.createTACS(self.ndof)
            self.nnodes = int(self.tacs.createNodeVec().getArray().size / 3)
        return self.tacs

    def get_ndv(self):
        return self.ndv

    def get_ndof(self):
        return self.ndof

    def get_nnodes(self):
        return self.nnodes

    def get_modal_sizes(self):
        return self.nmodes, self.nnodes*self.ndof

    def add_model_components(self,model,connection_srcs):
        model.add_subsystem('struct_modal_decomp',ModalDecomp(get_tacs = self.get_tacs,
                                                              get_ndv = self.get_ndv,
                                                              nmodes = self.nmodes))

        connection_srcs['x_s0'] = 'struct_modal_decomp.x_s0'
        connection_srcs['mode_shape'] = 'struct_modal_decomp.mode_shape'
        connection_srcs['modal_mass'] = 'struct_modal_decomp.modal_mass'
        connection_srcs['modal_stiffness'] = 'struct_modal_decomp.modal_stiffness'

    def add_scenario_components(self,model,scenario,connection_srcs):
        scenario.add_subsystem('struct_funcs',TacsFunctions(get_tacs=self.get_tacs,get_ndv=self.get_ndv,get_funcs=self.get_funcs,f5_writer=self.f5_writer))
        scenario.add_subsystem('struct_mass',TacsMass(get_tacs=self.get_tacs,get_ndv=self.get_ndv))

        connection_srcs['f_struct'] = scenario.name+'.struct_funcs.f_struct'
        connection_srcs['mass'] = scenario.name+'.struct_mass.mass'

    def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):

        struct = om.Group()
        struct.add_subsystem('modal_forces',ModalForces(get_modal_sizes=self.get_modal_sizes))
        struct.add_subsystem('modal_solver',ModalSolver(nmodes=self.nmodes))
        struct.add_subsystem('modal_disps',ModalDisplacements(get_modal_sizes=self.get_modal_sizes))

        fsi_group.add_subsystem('struct',struct)

        connection_srcs['mf']  = scenario.name+'.'+fsi_group.name+'.struct.modal_forces.mf'
        connection_srcs['z']   = scenario.name+'.'+fsi_group.name+'.struct.modal_solver.z'
        connection_srcs['u_s'] = scenario.name+'.'+fsi_group.name+'.struct.modal_disps.u_s'

    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):

        forces_path =  scenario.name+'.'+fsi_group.name+'.struct.modal_forces'
        solver_path =  scenario.name+'.'+fsi_group.name+'.struct.modal_solver'
        disps_path  =  scenario.name+'.'+fsi_group.name+'.struct.modal_disps'

        model.connect(connection_srcs['dv_struct'],'struct_modal_decomp.dv_struct')

        model.connect(connection_srcs['f_s'],[forces_path+'.f_s'])
        model.connect(connection_srcs['mf'],[solver_path+'.mf'])
        model.connect(connection_srcs['z'],[disps_path+'.z'])

        model.connect(connection_srcs['mode_shape'],forces_path+'.mode_shape')
        model.connect(connection_srcs['mode_shape'],disps_path+'.mode_shape')
        model.connect(connection_srcs['modal_stiffness'],[solver_path+'.k'])

        model.connect(connection_srcs['x_s0'],scenario.name+'.struct_funcs.x_s0')
        model.connect(connection_srcs['u_s'],scenario.name+'.struct_funcs.u_s')
        model.connect(connection_srcs['dv_struct'],scenario.name+'.struct_funcs.dv_struct')
        model.connect(connection_srcs['x_s0'],scenario.name+'.struct_mass.x_s0')
        model.connect(connection_srcs['dv_struct'],scenario.name+'.struct_mass.dv_struct')


class ModalDecomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('get_tacs', default = None, desc='function to get tacs')
        self.options.declare('get_ndv', default = None, desc='function to get number of design variables in tacs')
        self.options.declare('nmodes', default = 1, desc = 'number of modes to kept')
        self.options['distributed'] = True

    def setup(self):

        # TACS assembler setup
        self.tacs = self.options['get_tacs'](self.comm)
        self.ndv = self.options['get_ndv']()
        self.nmodes = self.options['nmodes']

        # create some TACS bvecs that will be needed later
        self.xpts  = self.tacs.createNodeVec()
        self.tacs.getNodes(self.xpts)

        self.vec  = self.tacs.createVec()

        # OpenMDAO setup
        node_size  =     self.xpts.getArray().size
        self.state_size = self.vec.getArray().size
        self.ndof = int(self.state_size / (node_size/3))

        self.add_input('dv_struct',shape=self.ndv, desc='structural design variables')

        self.add_output('mode_shape', shape=(self.nmodes,self.state_size), desc='structural mode shapes')
        self.add_output('modal_mass', shape=self.nmodes, desc='modal mass')
        self.add_output('modal_stiffness', shape=self.nmodes, desc='modal stiffness')
        self.add_output('x_s0', shape = node_size, desc = 'undeformed nodal coordinates')

    def compute(self,inputs,outputs):

        self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        kmat = self.tacs.createFEMat()
        self.tacs.assembleMatType(TACS.PY_STIFFNESS_MATRIX,kmat)
        pc = TACS.Pc(kmat)
        subspace = 100
        restarts = 2
        self.gmres = TACS.KSM(kmat, pc, subspace, restarts)

        # Guess for the lowest natural frequency
        sigma_hz = 1.0
        sigma = 2.0*np.pi*sigma_hz

        mmat = self.tacs.createFEMat()
        self.tacs.assembleMatType(TACS.PY_MASS_MATRIX,mmat)

        self.freq = TACS.FrequencyAnalysis(self.tacs, sigma, mmat, kmat, self.gmres,
                                      num_eigs=self.nmodes, eig_tol=1e-12)
        self.freq.solve()

        outputs['x_s0'] = self.xpts.getArray()
        for imode in range(self.nmodes):
            eig, err = self.freq.extractEigenvector(imode,self.vec)
            outputs['modal_mass'][imode] = 1.0
            outputs['modal_stiffness'][imode] = eig
            for idof in range(3):
                outputs['mode_shape'][imode,:] = self.vec.getArray()

            # debugging
            #matrix = np.zeros((int(self.xpts.getArray().size/3),6))
            #matrix[:,0] = self.xpts.getArray()[ ::3]
            #matrix[:,1] = self.xpts.getArray()[1::3]
            #matrix[:,2] = self.xpts.getArray()[2::3]
            #matrix[:,3] = self.vec.getArray()[ ::6]
            #matrix[:,4] = self.vec.getArray()[1::6]
            #matrix[:,5] = self.vec.getArray()[2::6]
            #np.savetxt('mode_shape'+str(imode)+'.dat',matrix)

class ModalSolver(om.ExplicitComponent):
    """
    Steady Modal structural solver
      K z - mf = 0
    """
    def initialize(self):
        self.options.declare('nmodes',default=1)
    def setup(self):
        nmodes = self.options['nmodes']
        self.add_input('k', shape=nmodes, val=np.ones(nmodes), desc = 'modal stiffness')
        self.add_input('mf', shape=nmodes, val=np.ones(nmodes), desc = 'modal force')

        self.add_output('z', shape=nmodes, val=np.ones(nmodes), desc = 'modal displacement')

    def compute(self,inputs,outputs):
        k = inputs['k']
        outputs['z'] = inputs['mf'] / inputs['k']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'z' in d_outputs:
                if 'mf' in d_inputs:
                    d_outputs['z'] += d_inputs['mf'] / inputs['k']
                if 'k' in d_inputs:
                    d_outputs['z'] += - inputs['mf'] / (inputs['k']**2.0) * d_inputs['k']
        if mode == 'rev':
            if 'z' in d_outputs:
                if 'mf' in d_inputs:
                    d_inputs['mf'] += d_outputs['z'] / inputs['k']
                if 'k' in d_inputs:
                    d_inputs['k'] += - inputs['mf'] / (inputs['k']**2.0) * d_outputs['z']

class ModalForces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('get_modal_sizes')

    def setup(self):
        self.nmodes, self.mode_size = self.options['get_modal_sizes']()

        self.add_input('mode_shape',shape=(self.nmodes,self.mode_size), desc='structural mode shapes')
        self.add_input('f_s',shape=self.mode_size,desc = 'nodal force')
        self.add_output('mf',shape=self.nmodes, desc = 'modal force')

    def compute(self,inputs,outputs):
        outputs['mf'][:] = 0.0
        for imode in range(self.nmodes):
            outputs['mf'][imode] = np.sum(inputs['mode_shape'][imode,:] * inputs['f_s'][:])

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'mf' in d_outputs:
                if 'f_s' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        d_outputs['mf'][imode] += np.sum(inputs['mode_shape'][imode,:] * d_inputs['f_s'][:])
        if mode=='rev':
            if 'mf' in d_outputs:
                if 'f_s' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        d_inputs['f_s'][:] += inputs['mode_shape'][imode,:] * d_outputs['mf'][imode]

class ModalDisplacements(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('get_modal_sizes')

    def setup(self):
        self.nmodes, self.mode_size = self.options['get_modal_sizes']()

        self.add_input('mode_shape',shape=(self.nmodes,self.mode_size), desc='structural mode shapes')
        self.add_input('z',shape=self.nmodes, desc = 'modal displacement')
        self.add_output('u_s',shape=self.mode_size,desc = 'nodal displacement')

    def compute(self,inputs,outputs):
        outputs['u_s'][:] = 0.0
        for imode in range(self.nmodes):
            outputs['u_s'][:] += inputs['mode_shape'][imode,:] * inputs['z'][imode]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'u_s' in d_outputs:
                if 'z' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        d_outputs['u_s'][:] += inputs['mode_shape'][imode,:] * d_inputs['z'][imode]
        if mode=='rev':
            if 'u_s' in d_outputs:
                if 'z' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        d_inputs['z'][imode] += np.sum(inputs['mode_shape'][imode,:] * d_outputs['u_s'][:])

