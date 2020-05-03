import numpy as np
from mpi4py import MPI

import openmdao.api as om
from rlt import SimpleLDTransfer

transfer_dtype = 'd'
# hard-coded ndof for aerodynamic solver
ndof_a = 3

class RLT_disp_xfer(om.ExplicitComponent):
    """
    Component to perform displacement transfer using RLT
    """
    def initialize(self):
        # Set options
        self.options.declare('xfer_object')
        self.options.declare('ndof_s')
        self.options.declare('nn_s')
        self.options.declare('nn_a')

        self.options['distributed'] = True

        # Flag used to prevent warning for fwd derivative d(u_a)/d(x_a0)
        self.check_partials = True

    def setup(self):

        # get the inputs
        RLT    = self.options['xfer_object']
        ndof_s = self.options['ndof_s']
        nn_s   = self.options['nn_s']
        nn_a   = self.options['nn_a']

        # get the isAero and isStruct flags from RLT python object
        # this is done to pseudo parallelize the modal solver, where
        # only the root proc does the computations.
        self.isAero = RLT.isAero
        self.isStruct = RLT.isStruct

        # set attributes
        self.transfer = RLT.transfer
        self.ndof_s = ndof_s
        self.nn_s = nn_s
        self.ndof_a = ndof_a
        self.nn_a = nn_a
        total_dof_struct = self.nn_s * self.ndof_s
        total_dof_aero = self.nn_a * self.ndof_a

        if self.isStruct:
            # RLT depends on TACS vector types.
            #   ustruct : holds the structural states
            #   struct_seed : used as a seed for structural displacements and forces
            self.tacs = RLT.structSolver.structure
            self.ustruct = self.tacs.createVec()
            self.struct_seed = self.tacs.createVec()
        else:
            self.ustruct = None

        # Get the source indices for each of the distributed inputs.
        irank = self.comm.rank

        ax_list = self.comm.allgather(total_dof_aero)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        sx_list = self.comm.allgather(total_dof_struct)
        sx1 = np.sum(sx_list[:irank])
        sx2 = np.sum(sx_list[:irank+1])

        su_list = self.comm.allgather(total_dof_struct)
        su1 = np.sum(su_list[:irank])
        su2 = np.sum(su_list[:irank+1])

        # Inputs
        self.add_input('x_a0', shape=total_dof_aero,
                       src_indices=np.arange(ax1, ax2, dtype=int),
                       desc='Initial aerodynamic surface node coordinates')
        self.add_input('x_s0', shape = total_dof_struct,
                       src_indices = np.arange(sx1, sx2, dtype=int),
                       desc='initial structural node coordinates')
        self.add_input('u_s', shape=total_dof_struct,
                       src_indices=np.arange(su1, su2, dtype=int),
                       desc='Structural node displacements')

        # Outputs
        self.add_output('u_a', shape=total_dof_aero,
                        val=np.zeros(total_dof_aero),
                        desc='Aerodynamic surface displacements')

        # TODO disable for now for the modal solver stuff.
        # Partials
        # self.declare_partials('u_a', ['x_a0','u_s'])

    def compute(self, inputs, outputs):
        # Update transfer object with the current set of CFD points
        self.transfer.setAeroSurfaceNodes(inputs['x_a0'])

        if self.isStruct:
            # Set the structural displacements
            ustruct_array = self.ustruct.getArray()
            ustruct_array[:] = inputs['u_s']
        self.transfer.setDisplacements(self.ustruct)

        # Get out the aerodynamic displacements
        self.transfer.getDisplacements(outputs['u_a'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # TODO check if the partial computations are okay when isStruct is not True on all procs
        if mode == 'fwd':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    if self.isStruct:
                        # Set the forward seed on the structural displacements
                        self.struct_seed.zeroEntries()
                        seed_array = self.struct_seed.getArray()
                        seed_array[:] = d_inputs['u_s']
                        self.transfer.setDisplacementPerturbation(self.struct_seed)

                    # Retrieve the seed from the aerodynamic displacements
                    u_ad = np.zeros(self.nn_a*self.ndof_a, dtype=transfer_dtype)
                    self.transfer.getAeroSurfacePerturbation(u_ad)
                    d_outputs['u_a'] += u_ad

                if 'x_a0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('Forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    if self.isStruct:
                        # Set the reverse seed from the aero displacements and
                        # retrieve the seed on the structural displacements.
                        # Note: Could also use setDisplacementsSens.
                        self.transfer.zeroReverseSeeds()
                        self.struct_seed.zeroEntries()
                        self.transfer.addAdjointDisplacements(d_outputs['u_a'], self.struct_seed)

                        # Pull the seed out of the TACS vector and accumulate
                        seed_array = self.struct_seed.getArray()
                        d_inputs['u_s'] += seed_array[:]

                if 'x_a0' in d_inputs:
                    # Set the reverse seed from the aero displacements
                    self.transfer.zeroReverseSeeds()
                    if self.isStruct:
                        self.transfer.setDisplacementsSens(self.ustruct,
                                                           self.struct_seed,
                                                           d_outputs['u_a'])

                    # Retrieve the seed on the aerodynamic surface nodes.
                    x_a0d = np.zeros(self.nn_a*self.ndof_a, dtype=transfer_dtype)
                    self.transfer.setAeroSurfaceNodesSens(x_a0d)
                    d_inputs['x_a0'] += x_a0d


class RLT_load_xfer(om.ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):
        # Set options
        self.options.declare('xfer_object')
        self.options.declare('ndof_s')
        self.options.declare('nn_s')
        self.options.declare('nn_a')

        self.options['distributed'] = True

        # Set everything we need to None before setup
        self.transfer = None
        self.tacs = None
        self.ndof_s = None
        self.ndof_a = None
        self.nn_s = None
        self.nn_a = None

        # Flag used to prevent warning for fwd derivative d(u_a)/d(x_a0)
        self.check_partials = True

    def setup(self):

        # get the inputs
        RLT    = self.options['xfer_object']
        ndof_s = self.options['ndof_s']
        nn_s   = self.options['nn_s']
        nn_a   = self.options['nn_a']

        # get the isAero and isStruct flags from RLT python object
        # this is done to pseudo parallelize the modal solver, where
        # only the root proc does the computations.
        self.isAero = RLT.isAero
        self.isStruct = RLT.isStruct

        # set attributes
        self.transfer = RLT.transfer
        self.ndof_s = ndof_s
        self.ndof_a = ndof_a
        self.nn_s = nn_s
        self.nn_a = nn_a
        total_dof_struct = self.nn_s * self.ndof_s
        total_dof_aero = self.nn_a * self.ndof_a

        if self.isStruct:
            # RLT depends on TACS vector types.
            #   fstruct : holds the forces on the structural nodes
            #   struct_seed : used as a seed for structural displacements and forces
            self.tacs = RLT.structSolver.structure
            self.fstruct = self.tacs.createVec()
            self.struct_seed = self.tacs.createVec()
        else:
            self.fstruct = None

        # Get the source indices for each of the distributed inputs.
        irank = self.comm.rank

        ax_list = self.comm.allgather(total_dof_aero)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        sx_list = self.comm.allgather(total_dof_struct)
        sx1 = np.sum(sx_list[:irank])
        sx2 = np.sum(sx_list[:irank+1])

        su_list = self.comm.allgather(total_dof_struct)
        su1 = np.sum(su_list[:irank])
        su2 = np.sum(su_list[:irank+1])

        # Inputs
        self.add_input('x_a0', shape=total_dof_aero,
                       src_indices=np.arange(ax1, ax2, dtype=int),
                       desc='Initial aerodynamic surface node coordinates')
        self.add_input('x_s0', shape = total_dof_struct,
                       src_indices = np.arange(sx1, sx2, dtype=int),
                       desc='initial structural node coordinates')
        self.add_input('u_s', shape=total_dof_struct,
                       src_indices=np.arange(su1, su2, dtype=int),
                       desc='Structural node displacements')

        self.add_input('f_a',  shape=total_dof_aero,
                       src_indices=np.arange(ax1, ax2, dtype=int),
                       desc='Aerodynamic force vector')

        # Outputs
        self.add_output('f_s', shape=total_dof_struct,
                        desc='structural force vector')

        # TODO disable for now for the modal solver stuff.
        # Partials
        # self.declare_partials('f_s', ['x_a0','f_a'])

    def compute(self, inputs, outputs):
        # Update transfer object with the current set of CFD points
        self.transfer.setAeroSurfaceNodes(inputs['x_a0'])

        if self.isStruct:
            # Set the aerodynamic forces and extract structural forces
            self.fstruct.zeroEntries()
        self.transfer.addAeroForces(inputs['f_a'], self.fstruct)

        if self.isStruct:
            # Get numpy array version of structural forces
            f_s = self.fstruct.getArray()
            outputs['f_s'] = -f_s[:] #This negative sign was necessary, not exactly sure why

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'f_s' in d_outputs:
                if 'f_a' in d_inputs:
                    # Set the forward seed on the aerodynamic forces and pull it
                    # out on struct_seed
                    self.struct_seed.zeroEntries()
                    self.transfer.addAeroForces(d_inputs['f_a'], self.struct_seed)
                    f_sd = self.struct_seed.getArray()
                    d_outputs['f_s'] -= f_sd[:]

                if 'x_a0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('Forward mode requested but not implemented')

        if mode == 'rev':
            if 'f_s' in d_outputs:
                # Set the reverse seed on the structural forces into the
                # struct_seed vector
                self.transfer.zeroReverseSeeds()
                self.struct_seed.zeroEntries()
                seed_array = self.struct_seed.getArray()
                seed_array[:] = d_outputs['f_s']

                if 'f_a' in d_inputs:
                    # Extract the reverse seed on the aerodynamic forces
                    f_ab = np.zeros(self.nn_a*self.ndof_a, dtype=transfer_dtype)
                    self.transfer.addAeroForcesSens(np.ravel(inputs['f_a']),
                                                             np.ravel(f_ab),
                                                             self.struct_seed)
                    d_inputs['f_a'] = -f_ab

                if 'x_a0' in d_inputs:
                    # Set up numpy arrays. We need the tmp array as a
                    # placeholder for unneeded data from addAeroForcesSens
                    x_a0d = np.zeros(self.nn_a*self.ndof_a, dtype=transfer_dtype)
                    tmp = np.zeros_like(x_a0d, dtype=transfer_dtype)

                    # Set the reverse seed
                    self.transfer.zeroReverseSeeds()
                    self.transfer.addAeroForcesSens(inputs['f_a'], tmp,
                                                    self.struct_seed)

                    # Pull it out on x_a0d
                    self.transfer.setAeroSurfaceNodesSens(x_a0d)
                    d_inputs['x_a0'] -= x_a0d

class RLT_builder(object):

    def __init__(self, options, aero_builder, struct_builder):
        self.options=options
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder

    # api level method for all builders
    def init_xfer_object(self, comm):

        aero_solver   = self.aero_builder.get_solver()
        struct_solver = self.struct_builder.get_solver()


        # if the struct_solver is none, we should pass none instead of dummy pytacs
        if struct_solver is None:
            dummy_pytacs = None
        else:
            #TODO: make this better. RLT is expecting a pytacs object to be
            # passed in, but it only needs the actual TACS object and the
            # structural comm. So I just created a dummy object for now that
            # references the attributes of the struct_assembler. I could have
            # passed in the struct_assembler, but wasn't sure if that was
            # bad practice.
            class dummy_pytacs: pass
            dummy_pytacs.structure = struct_solver
            dummy_pytacs.comm = comm

        self.xfer_object = SimpleLDTransfer(
            aero_solver,
            dummy_pytacs,
            comm=comm,
            options=self.options
        )

        # TODO also do the necessary calls to the struct and aero builders to fully initialize MELD
        # for now, just save the counts
        self.struct_ndof = self.struct_builder.get_ndof()
        self.struct_nnodes = self.struct_builder.get_nnodes()
        self.aero_nnodes = self.aero_builder.get_nnodes()

    # api level method for all builders
    def get_xfer_object(self):
        return self.xfer_object

    # api level method for all builders
    def get_element(self):

        disp_xfer = RLT_disp_xfer(
            xfer_object=self.xfer_object,
            ndof_s=self.struct_ndof,
            nn_s=self.struct_nnodes,
            nn_a=self.aero_nnodes,
        )

        load_xfer = RLT_load_xfer(
            xfer_object=self.xfer_object,
            ndof_s=self.struct_ndof,
            nn_s=self.struct_nnodes,
            nn_a=self.aero_nnodes,
        )

        return disp_xfer, load_xfer