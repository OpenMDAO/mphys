from .fsi_group import FsiSolver

from .tacs_component import TacsMesh, TacsSolver, TacsFunctions, TacsMassFunction
from .displacement_xfer_component import FuntofemDisplacementTransfer
from .load_xfer_component import FuntofemLoadTransfer

from tacs import TACS, functions
from funtofem import TransferScheme

#TODO: multiple FSI groups with shared or new solver objects

class FsiComps(object):
    """
    This class is design for adding steady FSI systems to an OpenMDAO model/group.
    It contains some setup callback functions so that TACS, ADFlow, and
    MELD objects can be shared between OpenMDAO components

    These helper functions are combined like this because transfer components
    needs to know some information (vector sizes and comms) from the disciplinary
    solver components

    """
    def __init__(self,tacs_setup,meld_setup):
        # Structural data to keep track of
        self.tacs_setup = tacs_setup
        self.struct_comm = None
        self.tacs = None
        self.struct_ndv = 0
        self.struct_ndof = 0
        self.struct_nprocs = 0

        # Transfer data to keep track of
        self.meld_setup = meld_setup
        self.xfer_ndof = 3
        self.meld = None

    def add_tacs_mesh(self,model,prefix='',reuse_solvers=True):
        # Initialize the disciplinary meshes
        tacs_mesh   = TacsMesh(tacs_mesh_setup=self.tacs_mesh_setup)
        model.add_subsystem('struct_mesh',tacs_mesh)

    def add_tacs_functions(self,model,prefix='',reuse_solvers=True):
        # Initialize the function evaluators
        struct_funcs,struct_mass = self._initialize_function_evaluators()
        model.add_subsystem('struct_funcs',struct_funcs)
        model.add_subsystem('struct_mass',struct_mass)

    def add_fsi_subsystems(self,model,aero,aero_nnodes,prefix='',reuse_solvers=True):
        self.aero_nnodes = aero_nnodes

        # Structural data to keep track of
        self.struct_nprocs = self.tacs_setup['nprocs']

        # Initialize the disciplinary solvers
        struct, disp_xfer, load_xfer = self._initialize_solvers()

        # Initialize the coupling group
        fsi_solver = FsiSolver(aero=aero,struct=struct,disp_xfer=disp_xfer,load_xfer=load_xfer,struct_nprocs=self.struct_nprocs,get_vector_size=self.get_aero_surface_size)

        model.add_subsystem(prefix+'fsi_solver',fsi_solver)

    def create_fsi_connections(self,model,struct_mesh='struct_mesh',aero_mesh='aero_mesh',fsi_solver='fsi_solver',
                                    nonlinear_xfer=False):

        model.connect(fsi_solver+'.struct.u_s',[fsi_solver+'.disp_xfer.u_s'])
        if nonlinear_xfer:
            model.connect(fsi_solver+'.struct.u_s',[fsi_solver+'.load_xfer.u_s'])

        model.connect(fsi_solver+'.disp_xfer.u_a',fsi_solver+'.geo_disps.u_a')
        model.connect(fsi_solver+'.geo_disps.x_a',fsi_solver+'.aero.deformer.x_a')
        model.connect(fsi_solver+'.aero.deformer.x_g',[fsi_solver+'.aero.solver.x_g',
                                                      fsi_solver+'.aero.forces.x_g'])
        model.connect(fsi_solver+'.aero.solver.q',[fsi_solver+'.aero.forces.q'])
        model.connect(fsi_solver+'.aero.forces.f_a',[fsi_solver+'.load_xfer.f_a'])
        model.connect(fsi_solver+'.load_xfer.f_s',fsi_solver+'.struct.f_s')

    def _initialize_mesh(self,reuse_solvers):
        """
        Initialize the different discipline meshes
        """

        return tacs_mesh 

    def _initialize_solvers(self):
        """
        Initialize the different disciplinary solvers
        """
        # Initialize the structural solver
        f5_writer = None
        if 'f5_writer' in self.tacs_setup:
            f5_writer = self.tacs_setup['f5_writer']
        struct    = TacsSolver(tacs_solver_setup=self.tacs_solver_setup,tacs_f5_writer=f5_writer)

        # Initialize the transfers
        disp_xfer   = FuntofemDisplacementTransfer(disp_xfer_setup=self.disp_xfer_setup)
        load_xfer   =         FuntofemLoadTransfer(load_xfer_setup=self.load_xfer_setup)

        return struct, disp_xfer, load_xfer

    def _initialize_function_evaluators(self):
        # Set up the structural solver
        struct_funcs = TacsFunctions(tacs_func_setup=self.tacs_func_setup)
        struct_mass = TacsMassFunction(tacs_func_setup=self.tacs_func_setup)
        return struct_funcs, struct_mass

    def tacs_mesh_setup(self,comm):
        """
        Setup callback function for TACS intial setup: reading the mesh and
        assigning elements
        """
        mesh_file        = self.tacs_setup['mesh_file']
        add_elements     = self.tacs_setup['add_elements']

        mesh = TACS.MeshLoader(comm)
        mesh.scanBDFFile(mesh_file)

        self.struct_ndof, self.struct_ndv = add_elements(mesh)

        self.tacs = mesh.createTACS(self.struct_ndof)

        self.struct_nnodes = int(self.tacs.createNodeVec().getArray().size / 3)
        return self.tacs

    def tacs_solver_setup(self,comm):
        """
        Setup callback function for TACS solver setup.
        """

        mat = self.tacs.createFEMat()
        pc = TACS.Pc(mat)

        self.mat = mat
        self.pc = pc

        subspace = 100
        restarts = 2
        gmres = TACS.KSM(mat, pc, subspace, restarts)

        return self.tacs, mat, pc, gmres, self.struct_ndv

    def tacs_func_setup(self,comm):
        """
        Setup callback function for TACS function evaluation.
        The user provides a list TACS function names (strings)
        """
        func_str_list = self.tacs_setup['func_list']
        func_list = []
        for func in func_str_list:
            if func.lower() == 'ks_failure':
                ksweight = 100.0
                func_list.append(functions.KSFailure(self.tacs, ksweight))

            elif func.lower() == 'compliance':
                func_list.append(functions.Compliance(self.tacs))

            elif func.lower() == 'mass':
                func_list.append(functions.StructuralMass(self.tacs))

        return func_list, self.tacs, self.struct_ndv, self.mat, self.pc

    def get_aero_surface_size(self):
        return self.aero_nnodes*3

    def disp_xfer_setup(self,comm):
        isym = self.meld_setup['isym']
        n    = self.meld_setup['n']
        beta = self.meld_setup['beta']

        # Assumption: The transfer comm is the same as the aero comm
        self.meld = TransferScheme.pyMELD(comm,comm,0,self.struct_comm,0,isym,n,beta)
        return self.meld, self.aero_nnodes, self.struct_nnodes, self.struct_ndof

    def load_xfer_setup(self):
        return self.meld, self.aero_nnodes, self.struct_nnodes, self.struct_ndof
