#!/usr/bin/env python
import numpy as np
from mpi4py import MPI
from tacs import TACS
import openmdao.api as om
from mphys import Builder
from mphys.mphys_tacs import TACSFuncsGroup

class ModalDecomp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('tacs_assembler', default = None, desc='tacs object')
        self.options.declare('ndv', default = 1, desc='number of design variables in tacs')
        self.options.declare('nmodes', default = 15, desc = 'number of modes to kept')

    def setup(self):
        self.tacs_assembler = self.options['tacs_assembler']
        self.ndv = self.options['ndv']
        self.nmodes = self.options['nmodes']

        if self.comm.rank == 0:
            self.xpts  = self.tacs_assembler.createNodeVec()
            self.tacs_assembler.getNodes(self.xpts)

            self.vec  = self.tacs_assembler.createVec()

            # OpenMDAO setup
            node_size  =     self.xpts.getArray().size
            self.state_size = self.vec.getArray().size
            self.ndof = int(self.state_size / (node_size/3))
        else:
            node_size = 0
            self.state_size = 0

        self.add_input('dv_struct', shape_by_conn=True, desc='structural design variables', tags=['mphys_input'])

        # instead of using 2d arrays for the mode shapes, we use a flattened array with modes back to back.
        # this is because in OpenMDAO version 2.10.1, the src_indices option for 2D arrays with empty i/o on
        # some procs is broken. Flattened works at least for the analysis.
        self.add_output('mode_shape', shape=self.nmodes*self.state_size,
                        distributed=True,
                        desc='structural mode shapes',
                        tags=['mphys_coupling'])

        output_shape = self.nmodes
        self.add_output('modal_mass', shape=output_shape, desc='modal mass', tags=['mphys_coupling'])
        self.add_output('modal_stiffness', shape=output_shape, desc='modal stiffness', tags=['mphys_coupling'])

        self.add_output('x_struct0', shape = node_size,
                        distributed = True,
                        desc = 'undeformed nodal coordinates', tags=['mphys_coodinates'])

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            self.tacs_assembler.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

            kmat = self.tacs_assembler.createFEMat()
            self.tacs_assembler.assembleMatType(TACS.PY_STIFFNESS_MATRIX,kmat,TACS.PY_NORMAL)
            pc = TACS.Pc(kmat)
            subspace = 100
            restarts = 2
            self.gmres = TACS.KSM(kmat, pc, subspace, restarts)

            # Guess for the lowest natural frequency
            sigma_hz = 1.0
            sigma = 2.0*np.pi*sigma_hz

            mmat = self.tacs_assembler.createFEMat()
            self.tacs_assembler.assembleMatType(TACS.PY_MASS_MATRIX,mmat,TACS.PY_NORMAL)

            self.freq = TACS.FrequencyAnalysis(self.tacs_assembler, sigma, mmat, kmat, self.gmres,
                                        num_eigs=self.nmodes, eig_tol=1e-12)
            self.freq.solve()

            outputs['x_struct0'] = self.xpts.getArray()
            for imode in range(self.nmodes):
                eig, err = self.freq.extractEigenvector(imode,self.vec)
                outputs['modal_mass'][imode] = 1.0
                outputs['modal_stiffness'][imode] = eig
                outputs['mode_shape'][imode*self.state_size:(imode+1)*self.state_size] = self.vec.getArray()

        outputs['modal_mass'] = self.comm.bcast(outputs['modal_mass'])
        outputs['modal_stiffness'] = self.comm.bcast(outputs['modal_stiffness'])

class ModalSolver(om.ExplicitComponent):
    """
    Steady modal structural solver
      K z - mf = 0
    """
    def initialize(self):
        self.options.declare('nmodes',default=1)
    def setup(self):

        nmodes = self.options['nmodes']

        self.add_input('modal_stiffness', shape_by_conn=True, desc = 'modal stiffness', tags=['mphys_coupling'])
        self.add_input('mf', shape_by_conn=True, desc = 'modal force')

        self.add_output('z', shape=nmodes,
                             val=np.ones(nmodes),
                             desc = 'modal displacement')

    def compute(self,inputs,outputs):
        k = inputs['modal_stiffness']
        outputs['z'] = inputs['mf'] / k

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
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

        self.add_input(
            'mode_shape',
            shape_by_conn=True,
            distributed=True,
            desc='structural mode shapes'
        )

        self.add_input('f_struct',shape_by_conn=True, distributed=True, desc = 'nodal force', tags=['mphys_coupling'])

        self.add_output('mf',shape=self.nmodes, desc = 'modal force')

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            outputs['mf'][:] = 0.0
            for imode in range(self.nmodes):
                outputs['mf'][imode] = np.sum(inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * inputs['f_struct'][:])
        outputs['mf'] = self.comm.bcast(outputs['mf'])

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'mf' in d_outputs:
                if 'f_struct' in d_inputs:
                    if self.comm.rank == 0:
                        for imode in range(self.options['nmodes']):
                            d_outputs['mf'][imode] += np.sum(inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_inputs['f_struct'][:])
                    d_outputs['mf'] = self.comm.bcast(d_outputs['mf'])
        if mode=='rev':
            if 'mf' in d_outputs:
                if 'f_struct' in d_inputs:
                    if self.comm.rank == 0:
                        for imode in range(self.options['nmodes']):
                            d_inputs['f_struct'][:] += inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_outputs['mf'][imode]

class ModalDisplacements(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes')
        self.options.declare('mode_size')

    def setup(self):
        self.nmodes    = self.options['nmodes']
        self.mode_size = self.options['mode_size']

        self.add_input('mode_shape', shape_by_conn=True, distributed=True, desc='structural mode shapes', tags=['mphys_coupling'])
        self.add_input('z', shape_by_conn=True, desc='modal displacement')

        self.add_output('u_struct', shape=self.mode_size, val = np.zeros(self.mode_size), desc = 'nodal displacement', tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        if self.comm.rank == 0:
            outputs['u_struct'][:] = 0.0
            for imode in range(self.nmodes):
                outputs['u_struct'][:] += inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * inputs['z'][imode]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'u_struct' in d_outputs:
                if 'z' in d_inputs:
                    if self.comm.rank == 0:
                        for imode in range(self.options['nmodes']):
                            d_outputs['u_struct'][:] += inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_inputs['z'][imode]
                if 'mode_shape' in d_inputs:
                    if self.comm.rank == 0:
                        for imode in range(self.options['nmodes']):
                            d_outputs['u_struct'][:] += d_inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * inputs['z'][imode]
        if mode=='rev':
            if 'u_struct' in d_outputs:
                if 'z' in d_inputs:
                    if self.comm.rank == 0:
                        for imode in range(self.options['nmodes']):
                            d_inputs['z'][imode] += np.sum(inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] * d_outputs['u_struct'][:])
                    d_inputs['z'] = self.comm.bcast(d_inputs['z'])
                if 'mode_shape' in d_inputs:
                    if self.comm.rank == 0:
                        for imode in range(self.options['nmodes']):
                            d_inputs['mode_shape'][imode*self.mode_size:(imode+1)*self.mode_size] += inputs['z'][imode] * d_outputs['u_struct'][:]

class ModalGroup(om.Group):
    def initialize(self):
        self.options.declare('nmodes')
        self.options.declare('nnodes')
        self.options.declare('ndof')
        self.options.declare('check_partials')

    def setup(self):
        nmodes = self.options['nmodes']
        mode_size = self.options['nnodes']*self.options['ndof']

        self.add_subsystem('modal_forces', ModalForces(
            nmodes=nmodes,
            mode_size=mode_size),
            promotes_inputs=['mode_shape','f_struct']
        )
        self.add_subsystem('modal_solver', ModalSolver(nmodes=nmodes),
            promotes_inputs=['modal_stiffness'])

        self.add_subsystem('modal_disps', ModalDisplacements(
            nmodes=nmodes,
            mode_size=mode_size),
            promotes_inputs=['mode_shape'],
            promotes_outputs=['u_struct']
        )

        self.connect('modal_forces.mf', 'modal_solver.mf')
        self.connect('modal_solver.z', 'modal_disps.z')

class ModalBuilder(Builder):
    def __init__(self, options, nmodes, include_tacs_functions = True, check_partials=False):
        self.options = options
        self.nmodes = nmodes
        self.check_partials = check_partials
        self.include_tacs_functions = include_tacs_functions

    def initialize(self, comm):
        self.comm = comm

        solver_dict={}

        if comm.rank == 0:
            mesh = TACS.MeshLoader(MPI.COMM_SELF)
            mesh.scanBDFFile(self.options['mesh_file'])

            ndof, ndv = self.options['add_elements'](mesh)

            assembler = mesh.createTACS(ndof)

            self.number_of_nodes = int(assembler.createNodeVec().getArray().size / 3)

            mat = assembler.createFEMat()
            pc = TACS.Pc(mat)

            subspace = 100
            restarts = 2
            gmres = TACS.KSM(mat, pc, subspace, restarts)

            solver_objects = [mat, pc, gmres, solver_dict]

        else:
            self.number_of_nodes = 0
            solver_objects = []
            assembler = None
            ndv  = 0
            ndof = 0

        self.ndv  = comm.bcast(ndv,  root=0)
        self.ndof = comm.bcast(ndof, root=0)
        if 'f5_writer' in self.options.keys():
            solver_objects[3]['f5_writer'] = self.options['f5_writer']
        solver_objects[3]['ndv'] = self.ndv
        solver_objects[3]['get_funcs'] = self.options['get_funcs']

        self.tacs_assembler = assembler
        self.solver_objects = solver_objects

    def get_coupling_group_subsystem(self):
        return ModalGroup(nmodes=self.nmodes,
                          nnodes=self.number_of_nodes,
                          ndof=self.ndof,
                          check_partials=self.check_partials)

    def get_mesh_coordinate_subsystem(self):
        return ModalDecomp(tacs_assembler=self.tacs_assembler,
                           ndv=self.ndv,
                           nmodes=self.nmodes)

    def get_post_coupling_subsystem(self):
        return TACSFuncsGroup(
            tacs_assembler=self.tacs_assembler,
            solver_objects=self.solver_objects,
            check_partials=True) # always want the function component to load the state in

    def get_ndof(self):
        return self.ndof

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def get_ndv(self):
        return self.ndv
