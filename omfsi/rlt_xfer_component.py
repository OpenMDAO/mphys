import numpy as np

from openmdao.api import ExplicitComponent
from rlt import SimpleLDTransfer
from omfsi import OmfsiAssembler

transfer_dtype = 'd'

class RLTAssembler(OmfsiAssembler):
    """
    Rigid link load and displacement transfer

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
        struct_ndof   = self.struct_assembler.get_ndof()
        struct_nnodes = self.struct_assembler.get_nnodes()
        aero_nnodes   = self.aero_assembler.get_nnodes()

        return RLT, struct_ndof, struct_nnodes, aero_nnodes

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
        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None

        # Flag used to prevent warning for fwd derivative d(u_a)/d(x_a0)
        self.check_partials = True

    def setup(self):
        RLT, struct_ndof, struct_nnodes, aero_nnodes = self.options['setup_function'](self.comm)

        self.transfer = RLT.transfer
        self.struct_ndof = struct_ndof
        self.struct_nnodes = struct_nnodes
        self.aero_nnodes = aero_nnodes

        self.tacs = RLT.structSolver.structure
        self.ustruct = self.tacs.createVec()
        self.struct_seed = self.tacs.createVec()

        irank = self.comm.rank

        ax_list = self.comm.allgather(aero_nnodes*3)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        sx_list = self.comm.allgather(struct_nnodes*3)
        sx1 = np.sum(sx_list[:irank])
        sx2 = np.sum(sx_list[:irank+1])

        su_list = self.comm.allgather(struct_nnodes*struct_ndof)
        su1 = np.sum(su_list[:irank])
        su2 = np.sum(su_list[:irank+1])

        # inputs
        # self.add_input('x_s0', shape=struct_nnodes*3,
        #                 src_indices=np.arange(sx1, sx2, dtype=int),
        #                 desc='initial structural node coordinates')
        self.add_input('x_a0', shape=aero_nnodes*3,
                        src_indices=np.arange(ax1, ax2, dtype=int),
                        desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s', shape=struct_nnodes*struct_ndof,
                        src_indices=np.arange(su1, su2, dtype=int),
                        desc='structural node displacements')

        # outputs
        self.add_output('u_a', shape=aero_nnodes*3, val=np.zeros(aero_nnodes*3), desc='aerodynamic surface displacements')

        # partials
        self.declare_partials('u_a', ['x_a0','u_s'])

    def compute(self, inputs, outputs):
        x_a0 = np.array(inputs['x_a0'], dtype=transfer_dtype)
        u_s  = np.array(inputs['u_s'], dtype=transfer_dtype)
        u_a  = np.array(outputs['u_a'], dtype=transfer_dtype)

        # Update transfer object with the current set of CFD points
        self.transfer.setAeroSurfaceNodes(np.ravel(x_a0))

        ustruct_array = self.ustruct.getArray()
        ustruct_array[:] = u_s

        self.transfer.setDisplacements(self.ustruct)
        self.transfer.getDisplacements(np.ravel(u_a))

        outputs['u_a'] = u_a

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """

        """
        if mode == 'fwd':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    d_in = d_inputs['u_s']
                    self.struct_seed.zeroEntries()
                    seed_array = self.struct_seed.getArray()
                    seed_array[:] = d_in

                    prod = np.zeros(self.aero_nnodes*3,dtype=transfer_dtype)

                    self.transfer.setDisplacements(self.struct_seed)
                    self.transfer.getDisplacements(np.ravel(prod))

                    d_outputs['u_a'] += np.array(prod, dtype=float)

                if 'x_a0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('Forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_a' in d_outputs:
                du_a = np.array(d_outputs['u_a'],dtype=transfer_dtype)
                if 'u_s' in d_inputs:
                    self.struct_seed.zeroEntries()
                    self.transfer.addAdjointDisplacements(np.ravel(du_a), self.struct_seed)
                    # Could also use setDisplacementsSens

                    seed_array = self.struct_seed.getArray()
                    d_inputs['u_s'] += seed_array[:]

                if 'x_a0' in d_inputs:
                    self.transfer.zeroReverseSeeds()
                    self.transfer.setDisplacementsSens(self.ustruct, self.struct_seed, np.ravel(du_a))
                    prod = np.zeros(d_inputs['x_a0'].size, dtype=transfer_dtype)
                    self.transfer.setAeroSurfaceNodesSens(np.ravel(prod))
                    d_inputs['x_a0'] += np.array(prod,dtype=float)


class RLTLoadTransfer(ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):
        self.options.declare('setup_function', desc='function to get shared data')

        self.options['distributed'] = True

        self.transfer = None
        self.tacs = None
        self.initialized_meld = False

        self.struct_ndof = None
        self.struct_nnodes = None
        self.aero_nnodes = None

        self.check_partials = True

    def setup(self):
        # get the transfer scheme object
        RLT, struct_ndof, struct_nnodes, aero_nnodes = self.options['setup_function'](self.comm)

        self.transfer = RLT.transfer
        self.struct_ndof = struct_ndof
        self.struct_nnodes = struct_nnodes
        self.aero_nnodes = aero_nnodes

        self.tacs = RLT.structSolver.structure
        self.fstruct = self.tacs.createVec()
        self.struct_seed = self.tacs.createVec()

        irank = self.comm.rank

        ax_list = self.comm.allgather(aero_nnodes*3)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        sx_list = self.comm.allgather(struct_nnodes*3)
        sx1 = np.sum(sx_list[:irank])
        sx2 = np.sum(sx_list[:irank+1])

        su_list = self.comm.allgather(struct_nnodes*struct_ndof)
        su1 = np.sum(su_list[:irank])
        su2 = np.sum(su_list[:irank+1])

        # inputs
        self.add_input('x_a0', shape=aero_nnodes*3,
                        src_indices=np.arange(ax1, ax2, dtype=int),
                        desc='initial aerodynamic surface node coordinates')
        self.add_input('f_a',  shape=aero_nnodes*3,
                        src_indices=np.arange(ax1, ax2, dtype=int),
                        desc='aerodynamic force vector')

        # outputs
        self.add_output('f_s', shape=struct_nnodes*struct_ndof,
                        desc='structural force vector')

        # partials
        self.declare_partials('f_s', ['x_a0','f_a'])

    def compute(self, inputs, outputs):
        x_a0 = np.array(inputs['x_a0'], dtype=transfer_dtype)
        f_a = np.array(inputs['f_a'], dtype=transfer_dtype)

        # Update transfer object with the current set of CFD points
        self.transfer.setAeroSurfaceNodes(np.ravel(x_a0))

        self.fstruct.zeroEntries()
        self.transfer.addAeroForces(np.ravel(f_a), self.fstruct)

        f_s = self.fstruct.getArray()
        outputs['f_s'] = -f_s[:] #TODO: we need to formalize how this force should be passed

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """

        """

        if mode == 'fwd':
            if 'f_s' in d_outputs:
                if 'f_a' in d_inputs:
                    df_a = np.array(d_inputs['f_a'],dtype=transfer_dtype)
                    self.struct_seed.zeroEntries()
                    self.transfer.addAeroForces(np.ravel(df_a), self.struct_seed)
                    seed_array = self.struct_seed.getArray()
                    d_outputs['f_s'] -= seed_array[:]

                if 'x_a0' in d_inputs:
                    if self.check_partials:
                        pass
                    else:
                        raise ValueError('Forward mode requested but not implemented')

        if mode == 'rev':
            if 'f_s' in d_outputs:
                self.transfer.zeroReverseSeeds()
                f_sb = np.array(d_outputs['f_s'], dtype=transfer_dtype)
                self.struct_seed.zeroEntries()
                seed_array = self.struct_seed.getArray()
                seed_array[:] = f_sb

                if 'f_a' in d_inputs:
                    f_ab = np.zeros(self.aero_nnodes*3, dtype=transfer_dtype)
                    self.transfer.addAeroForcesSens(np.ravel(inputs['f_a']),
                                            np.ravel(f_ab), self.struct_seed)
                    d_inputs['f_a'] = -f_ab

                if 'x_a0' in d_inputs:
                    prod = np.zeros(self.aero_nnodes*3,dtype=transfer_dtype)

                    self.transfer.zeroReverseSeeds()
                    tmp = np.zeros_like(prod)
                    self.transfer.addAeroForcesSens(np.ravel(inputs['f_a']), np.ravel(tmp), self.struct_seed)
                    self.transfer.setAeroSurfaceNodesSens(np.ravel(prod))
                    d_inputs['x_a0'] -= np.array(prod, dtype=float)

