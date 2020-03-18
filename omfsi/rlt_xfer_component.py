import numpy as np

from openmdao.api import ExplicitComponent
from rlt import SimpleLDTransfer
from omfsi import OmfsiAssembler

transfer_dtype = 'd'

class RLTAssembler(OmfsiAssembler):
    """
    Rigid link load and displacement transfer

    Sizes
        nn_a : number of aerodynamic surface nodes
        nn_s : number of structural nodes
        ndof_a : number of aerodynamic degrees of freedom (spatial)
        ndof_s : number of structural degrees of freedom

    Variables
        x_s0 : structural node coordinates (pre-deformation)
        x_a0 : aerodynamic surface node coordinates (pre-deformation)

        u_s : structural nodal displacements
        u_a : aerodynamic surface nodal displacements

        f_s : structural nodal forces
        f_a : aerodynamic surface nodal forces
    """

    def __init__(self, options, struct_assembler, aero_assembler):

        # transfer scheme options
        self.transferOptions = {
            'transfergaussorder': options['transfergaussorder'],
        }

        self.struct_assembler = struct_assembler
        self.aero_assembler = aero_assembler
        self.RLT = None

    def _get_transfer(self):
        if self.RLT is None:
            #TODO: make this better. RLT is expecting a pytacs object to be
            # passed in, but it only needs the actual TACS object and the
            # structural comm. So I just created a dummy object for now that
            # references the attributes of the struct_assembler. I could have
            # passed in the struct_assembler, but wasn't sure if that was
            # bad practice.
            class dummy_pytacs: pass
            dummy_pytacs.structure = self.struct_assembler.tacs
            dummy_pytacs.comm = self.struct_assembler.comm
            self.RLT = SimpleLDTransfer(self.aero_assembler.solver,
                                        dummy_pytacs,
                                        comm=self.comm,
                                        options=self.transferOptions)

        return self.RLT

    def add_model_components(self, model, connection_srcs):
        pass

    def add_scenario_components(self, model, scenario, connection_srcs):
        pass

    def add_fsi_components(self, model, scenario, fsi_group, connection_srcs):
        # Add displacement transfer component
        fsi_group.add_subsystem(
            'disp_xfer',
            RLTDisplacementTransfer(setup_function=self.xfer_setup))

        # Add load transfer component
        fsi_group.add_subsystem(
            'load_xfer',
            RLTLoadTransfer(setup_function=self.xfer_setup))

        # Connect variables
        base_name = scenario.name + '.' + fsi_group.name
        connection_srcs['u_a'] = base_name + '.disp_xfer.u_a'
        connection_srcs['f_s'] = base_name + '.load_xfer.f_s'

    def connect_inputs(self, model, scenario, fsi_group, connection_srcs):
        # Make connections between components
        base_name = scenario.name+'.'+fsi_group.name
        model.connect(connection_srcs['u_s'], [base_name + '.disp_xfer.u_s'])
        model.connect(connection_srcs['f_a'], [base_name + '.load_xfer.f_a'])
        model.connect(
            connection_srcs['x_a0'],
            [base_name + '.disp_xfer.x_a0', base_name + '.load_xfer.x_a0'])

    def xfer_setup(self, comm):
        # We want the displacement transfer and the load transfer components to
        # use the same RLT transfer class. So we create one RLT transfer object
        # and then pass it to the displacement and load transfer components
        # when they ask for it.
        self.comm = comm
        self.struct_assembler.comm = comm
        RLT = self._get_transfer()
        ndof_s = self.struct_assembler.get_ndof()
        nn_s = self.struct_assembler.get_nnodes()
        ndof_a = 3 #TODO: currently hard-coded, but maybe should be generalized
        nn_a = self.aero_assembler.get_nnodes()

        return RLT, ndof_s, nn_s, ndof_a, nn_a

class RLTDisplacementTransfer(ExplicitComponent):
    """
    Component to perform displacement transfer using RLT
    """
    def initialize(self):
        # Set options
        self.options.declare('setup_function', desc='function to get shared data')
        self.options['distributed'] = True

        # Set everything we need to None before setup
        self.tacs = None
        self.transfer = None
        self.ndof_s = None
        self.nn_s = None
        self.ndof_a = None
        self.nn_a = None

        # Flag used to prevent warning for fwd derivative d(u_a)/d(x_a0)
        self.check_partials = True

    def setup(self):
        RLT, ndof_s, nn_s, ndof_a, nn_a = self.options['setup_function'](self.comm)

        self.transfer = RLT.transfer
        self.ndof_s = ndof_s
        self.nn_s = nn_s
        self.ndof_a = ndof_a
        self.nn_a = nn_a
        total_dof_struct = self.nn_s * self.ndof_s
        total_dof_aero = self.nn_a * self.ndof_a

        # RLT depends on TACS vector types.
        #   ustruct : holds the structural states
        #   struct_seed : used as a seed for structural displacements and forces
        self.tacs = RLT.structSolver.structure
        self.ustruct = self.tacs.createVec()
        self.struct_seed = self.tacs.createVec()

        # Get the source indices for each of the distributed inputs.
        irank = self.comm.rank

        ax_list = self.comm.allgather(total_dof_aero)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        su_list = self.comm.allgather(total_dof_struct)
        su1 = np.sum(su_list[:irank])
        su2 = np.sum(su_list[:irank+1])

        # Inputs
        self.add_input('x_a0', shape=total_dof_aero,
                       src_indices=np.arange(ax1, ax2, dtype=int),
                       desc='Initial aerodynamic surface node coordinates')
        self.add_input('u_s', shape=total_dof_struct,
                       src_indices=np.arange(su1, su2, dtype=int),
                       desc='Structural node displacements')

        # Outputs
        self.add_output('u_a', shape=total_dof_aero,
                        val=np.zeros(total_dof_aero),
                        desc='Aerodynamic surface displacements')

        # Partials
        self.declare_partials('u_a', ['x_a0','u_s'])

    def compute(self, inputs, outputs):
        # Update transfer object with the current set of CFD points
        self.transfer.setAeroSurfaceNodes(inputs['x_a0'])

        # Set the structural displacements
        ustruct_array = self.ustruct.getArray()
        ustruct_array[:] = inputs['u_s']
        self.transfer.setDisplacements(self.ustruct)

        # Get out the aerodynamic displacements
        self.transfer.getDisplacements(outputs['u_a'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
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
                    # Set the reverse seed from the aero displacements and
                    # retrieve the seed on the structural displacements.
                    # Note: Could also use setDisplacementsSens.
                    self.transfer.zeroReverseSeeds()
                    self.struct_seed.zeroEntries()
                    self.transfer.addAdjointDisplacements(d_outputs['u_a'],
                                                          self.struct_seed)

                    # Pull the seed out of the TACS vector and accumulate
                    seed_array = self.struct_seed.getArray()
                    d_inputs['u_s'] += seed_array[:]

                if 'x_a0' in d_inputs:
                    # Set the reverse seed from the aero displacements
                    self.transfer.zeroReverseSeeds()
                    self.transfer.setDisplacementsSens(self.ustruct,
                                                       self.struct_seed,
                                                       d_outputs['u_a'])

                    # Retrieve the seed on the aerodynamic surface nodes.
                    x_a0d = np.zeros(self.nn_a*self.ndof_a, dtype=transfer_dtype)
                    self.transfer.setAeroSurfaceNodesSens(x_a0d)
                    d_inputs['x_a0'] += x_a0d


class RLTLoadTransfer(ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):
        # Set options
        self.options.declare('setup_function', desc='function to get shared data')
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
        # get the transfer scheme object
        RLT, ndof_s, nn_s, ndof_a, nn_a = self.options['setup_function'](self.comm)

        self.transfer = RLT.transfer
        self.ndof_s = ndof_s
        self.ndof_a = ndof_a
        self.nn_s = nn_s
        self.nn_a = nn_a
        total_dof_struct = self.nn_s * self.ndof_s
        total_dof_aero = self.nn_a * self.ndof_a

        # RLT depends on TACS vector types.
        #   fstruct : holds the forces on the structural nodes
        #   struct_seed : used as a seed for structural displacements and forces
        self.tacs = RLT.structSolver.structure
        self.fstruct = self.tacs.createVec()
        self.struct_seed = self.tacs.createVec()

        # Get the source indices for each of the distributed inputs.
        irank = self.comm.rank

        ax_list = self.comm.allgather(total_dof_aero)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        # Inputs
        self.add_input('x_a0', shape=total_dof_aero,
                       src_indices=np.arange(ax1, ax2, dtype=int),
                       desc='Initial aerodynamic surface node coordinates')
        self.add_input('f_a',  shape=total_dof_aero,
                       src_indices=np.arange(ax1, ax2, dtype=int),
                       desc='Aerodynamic force vector')

        # Outputs
        self.add_output('f_s', shape=total_dof_struct,
                        desc='structural force vector')

        # Partials
        self.declare_partials('f_s', ['x_a0','f_a'])

    def compute(self, inputs, outputs):
        # Update transfer object with the current set of CFD points
        self.transfer.setAeroSurfaceNodes(inputs['x_a0'])

        # Set the aerodynamic forces and extract structural forces
        self.fstruct.zeroEntries()
        self.transfer.addAeroForces(inputs['f_a'], self.fstruct)

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

