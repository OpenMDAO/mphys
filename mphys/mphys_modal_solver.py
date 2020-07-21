#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from mpi4py import MPI
from tacs import TACS, elements, functions
import openmdao.api as om

class ModalDecomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('struct_solver', default = None, desc='tacs object')
        self.options.declare('ndv', default = 1, desc='number of design variables in tacs')
        self.options.declare('nmodes', default = 15, desc = 'number of modes to kept')
        self.options['distributed'] = True

    def setup(self):

        # TACS assembler setup
        self.tacs = self.options['struct_solver']
        self.ndv = self.options['ndv']
        self.nmodes = self.options['nmodes']

        if self.comm.rank == 0:
            # create some TACS bvecs that will be needed later
            self.xpts  = self.tacs.createNodeVec()
            self.tacs.getNodes(self.xpts)

            self.vec  = self.tacs.createVec()

            # OpenMDAO setup
            node_size  =     self.xpts.getArray().size
            self.state_size = self.vec.getArray().size
            self.ndof = int(self.state_size / (node_size/3))
        else:
            node_size = 0
            self.state_size = 0

        self.add_input('dv_struct',shape=self.ndv, desc='structural design variables')

        # instead of using 2d arrays for the mode shapes, we use a flattened array with modes back to back.
        # this is because in OpenMDAO version 2.10.1, the src_indices option for 2D arrays with empty i/o on
        # some procs is broken. Flattened works at least for the analysis.
        self.add_output('mode_shape', shape=self.nmodes*self.state_size, desc='structural mode shapes')

        if self.comm.rank == 0:
            output_shape = self.nmodes
        else:
            output_shape = 0

        self.add_output('modal_mass', shape=output_shape, desc='modal mass')
        self.add_output('modal_stiffness', shape=output_shape, desc='modal stiffness')

        self.add_output('x_s0', shape = node_size, desc = 'undeformed nodal coordinates')

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

            kmat = self.tacs.createFEMat()
            self.tacs.assembleMatType(TACS.PY_STIFFNESS_MATRIX,kmat,TACS.PY_NORMAL)
            pc = TACS.Pc(kmat)
            subspace = 100
            restarts = 2
            self.gmres = TACS.KSM(kmat, pc, subspace, restarts)

            # Guess for the lowest natural frequency
            sigma_hz = 1.0
            sigma = 2.0*np.pi*sigma_hz

            mmat = self.tacs.createFEMat()
            self.tacs.assembleMatType(TACS.PY_MASS_MATRIX,mmat,TACS.PY_NORMAL)

            self.freq = TACS.FrequencyAnalysis(self.tacs, sigma, mmat, kmat, self.gmres,
                                        num_eigs=self.nmodes, eig_tol=1e-12)
            self.freq.solve()

            outputs['x_s0'] = self.xpts.getArray()
            for imode in range(self.nmodes):
                eig, err = self.freq.extractEigenvector(imode,self.vec)
                outputs['modal_mass'][imode] = 1.0
                outputs['modal_stiffness'][imode] = eig
                for idof in range(3):
                    # put every mode back to back instead of using a 2d array bec. the pseudo parallelism breaks that way... This should be fixed in OM and we can go back to using 2D arrays
                    outputs['mode_shape'][imode*self.state_size:(imode+1)*self.state_size] = self.vec.getArray()

class ModalSolver(om.ExplicitComponent):
    """
    Steady Modal structural solver
      K z - mf = 0
    """
    def initialize(self):
        self.options.declare('nmodes',default=1)
        self.options['distributed'] = True
    def setup(self):

        # adjust the indices for procs
        if self.comm.rank == 0:
            nmodes = self.options['nmodes']
            src_indices = np.arange(0, nmodes, dtype=int)
            input_shape = nmodes
        else:
            nmodes = 0
            src_indices = np.zeros(0)
            input_shape=0

        self.add_input('modal_stiffness', shape=input_shape,
                                          src_indices=src_indices,
                                          val=np.ones(nmodes), desc = 'modal stiffness')
        self.add_input('mf', shape=input_shape,
                             src_indices=src_indices,
                             val=np.ones(nmodes),
                             desc = 'modal force')

        self.add_output('z', shape=nmodes,
                             val=np.ones(nmodes),
                             desc = 'modal displacement')

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            k = inputs['modal_stiffness']
            outputs['z'] = inputs['mf'] / k

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if self.comm.rank == 0:
            k = inputs['modal_stiffness']
            if mode == 'fwd':
                if 'z' in d_outputs:
                    if 'mf' in d_inputs:
                        d_outputs['z'] += d_inputs['mf'] / k
                    if 'modal_stiffness' in d_inputs:
                        d_outputs['z'] += - inputs['mf'] / (k**2.0) * d_inputs['modal_stiffness']
            if mode == 'rev':
                if 'z' in d_outputs:
                    if 'mf' in d_inputs:
                        d_inputs['mf'] += d_outputs['z'] / k
                    if 'modal_stiffness' in d_inputs:
                        d_inputs['modal_stiffness'] += - inputs['mf'] / (k**2.0) * d_outputs['z']

class ModalForces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes')
        self.options.declare('mode_size')

    def setup(self):
        self.nmodes    = self.options['nmodes']
        self.mode_size = self.options['mode_size']

        # adjust the indices for procs
        if self.comm.rank == 0:
            src_indices = np.arange(0, self.nmodes*self.mode_size, dtype=int)
            input_shape = self.nmodes*self.mode_size
        else:
            src_indices = np.zeros(0)
            input_shape=0

        self.add_input(
            'mode_shape',
            shape=input_shape,
            src_indices=src_indices,
            desc='structural mode shapes'
        )

        if self.comm.rank == 0:
            src_indices = np.arange(0, self.mode_size, dtype=int)
            input_shape = self.mode_size
        else:
            src_indices = np.zeros((0))
            input_shape=0

        self.add_input('f_s',shape=input_shape, src_indices=src_indices,desc = 'nodal force')

        self.add_output('mf',shape=self.nmodes, desc = 'modal force')

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            outputs['mf'][:] = 0.0
            for imode in range(self.nmodes):
                outputs['mf'][imode] = np.sum(inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * inputs['f_s'][:])

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if self.comm.rank == 0:
            if mode=='fwd':
                if 'mf' in d_outputs:
                    if 'f_s' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_outputs['mf'][imode] += np.sum(inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_inputs['f_s'][:])
            if mode=='rev':
                if 'mf' in d_outputs:
                    if 'f_s' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_inputs['f_s'][:] += inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_outputs['mf'][imode]

class ModalDisplacements(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes')
        self.options.declare('mode_size')

    def setup(self):
        self.nmodes    = self.options['nmodes']
        self.mode_size = self.options['mode_size']

        # adjust the indices for procs
        if self.comm.rank == 0:
            src_indices = np.arange(0, self.nmodes*self.mode_size, dtype=int)
            input_shape = self.nmodes*self.mode_size
        else:
            src_indices = np.zeros(0)
            input_shape=0

        self.add_input(
            'mode_shape',
            shape=input_shape,
            src_indices=src_indices,
            desc='structural mode shapes'
        )

        if self.comm.rank == 0:
            src_indices = np.arange(0, self.nmodes, dtype=int)
            input_shape = self.nmodes
        else:
            src_indices = np.zeros(0)
            input_shape = 0

        self.add_input('z',shape=input_shape, src_indices=src_indices, desc='modal displacement')

        # its important that we set this to zero since this displacement value is used for the first iteration of the aero
        self.add_output('u_s', shape=self.mode_size, val = np.zeros(self.mode_size), desc = 'nodal displacement')

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            outputs['u_s'][:] = 0.0
            for imode in range(self.nmodes):
                outputs['u_s'][:] += inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * inputs['z'][imode]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if self.comm.rank == 0:
            if mode=='fwd':
                if 'u_s' in d_outputs:
                    if 'z' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_outputs['u_s'][:] += inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_inputs['z'][imode]
            if mode=='rev':
                if 'u_s' in d_outputs:
                    if 'z' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_inputs['z'][imode] += np.sum(inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_outputs['u_s'][:])

class ModalGroup(om.Group):
    def initialize(self):
        self.options.declare('solver')
        self.options.declare('solver_objects')
        self.options.declare('nmodes')
        self.options.declare('nnodes')
        self.options.declare('ndof')
        self.options.declare('check_partials')
        self.options.declare('as_coupling')

    def setup(self):
        nmodes = self.options['nmodes']
        mode_size = self.options['nnodes']*self.options['ndof']

        self.add_subsystem('modal_forces', ModalForces(
            nmodes=nmodes,
            mode_size=mode_size),
            promotes_inputs=['mode_shape','f_s']
        )
        self.add_subsystem('modal_solver', ModalSolver(nmodes=nmodes),
            promotes_inputs=['modal_stiffness'])

        self.add_subsystem('modal_disps', ModalDisplacements(
            nmodes=nmodes,
            mode_size=mode_size),
            promotes_inputs=['mode_shape'],
            promotes_outputs=['u_s']
        )

#        self.add_subsystem('funcs', TacsFunctions(
#            struct_solver=self.struct_solver,
#            struct_objects=self.struct_objects,
#            check_partials=self.check_partials),
#            promotes_inputs=['x_s0', 'dv_struct']
#        )
#
#        self.add_subsystem('mass', TacsMass(
#            struct_solver=self.struct_solver,
#            struct_objects=self.struct_objects,
#            check_partials=self.check_partials),
#            promotes_inputs=['x_s0', 'dv_struct']
#        )

    def configure(self):
        if self.comm.rank == 0:
            self.connect('modal_forces.mf', 'modal_solver.mf')
            self.connect('modal_solver.z', 'modal_disps.z')

class ModalBuilder(object):

    def __init__(self, options,nmodes=15,check_partials=False):
        self.options = options
        self.nmodes = nmodes
        self.check_partials = check_partials

    # api level method for all builders
    def init_solver(self, comm):

        # save the comm bec. we will use this to control parallel output sizes
        self.comm = comm

        solver_dict={}

        # only root proc will do useful stuff
        if comm.rank == 0:
            mesh = TACS.MeshLoader(MPI.COMM_SELF)
            mesh.scanBDFFile(self.options['mesh_file'])

            ndof, ndv = self.options['add_elements'](mesh)

            tacs = mesh.createTACS(ndof)

            nnodes = int(tacs.createNodeVec().getArray().size / 3)

            mat = tacs.createFEMat()
            pc = TACS.Pc(mat)

            subspace = 100
            restarts = 2
            gmres = TACS.KSM(mat, pc, subspace, restarts)

            solver_dict['nnodes'] = nnodes
            solver_dict['get_funcs'] = self.options['get_funcs']

            # check if the user provided a load function
            if 'load_function' in self.options:
                solver_dict['load_function'] = self.options['load_function']

            # put the rest of the stuff in a tuple
            solver_objects = [mat, pc, gmres, solver_dict]

        # on other procs, we just place dummy data
        else:
            solver_dict['nnodes'] = 0
            solver_objects = np.zeros((0))
            tacs = None
            ndv  = 0
            ndof = 0

        # ndv and ndof should be same on all procs
        ndv  = comm.bcast(ndv,  root=0)
        ndof = comm.bcast(ndof, root=0)
        solver_dict['ndv']  = ndv
        solver_dict['ndof'] = ndof

        # set the attributes on all procs
        self.solver_dict=solver_dict
        self.solver = tacs
        self.solver_objects = solver_objects

    # api level method for all builders
    def get_solver(self):
        return self.solver

    # api level method for all builders
    def get_element(self, **kwargs):
        return ModalGroup(solver=self.solver, solver_objects=self.solver_objects,
                          nmodes=self.nmodes,
                          nnodes=self.solver_dict['nnodes'],
                          ndof=self.solver_dict['ndof'],
                          check_partials=self.check_partials, **kwargs)

    def get_mesh_element(self):
        return ModalDecomp(struct_solver=self.solver,
                           ndv=self.solver_dict['ndv'],
                           nmodes=self.nmodes)

    def get_mesh_connections(self):
        return {
            # because we dont have a solver or funcs key,
            # mphys just assume that these will be connected
            # to the solver.
            'modal_stiffness': 'modal_stiffness',
            'mode_shape': 'mode_shape',
        }

    def get_ndof(self):
        return self.solver_dict['ndof']

    def get_nnodes(self):
        return self.solver_dict['nnodes']

    def get_ndv(self):
        return self.solver_dict['ndv']
