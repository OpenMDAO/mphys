from fsi_group import FsiSolver

from tacs_component import TacsSolver, StructuralGroup, TacsFunctions
from aero_group import AeroSolver
from displacement_xfer_component import FuntofemDisplacementTransfer
from load_xfer_component import FuntofemLoadTransfer

#TODO: multiple FSI groups with shared or new solver objects

class FsiComps(object):
    """
    This class is design for adding steady FSI systems to an OpenMDAO model/group.
    It contains some setup callback functions so that TACS, ADFlow, and
    MELD objects can be shared between OpenMDAO components

    These helpper functions are combined like this because transfer components
    needs to know some information (vector sizes and comms) from the disciplinary
    solver components

    """
    def __init__(self):
        # Flow data to keep track of
        self.adflow = None

        # Structural data to keep track of
        self.struct_comm = None
        self.tacs = None
        self.struct_ndv = 0
        self.struct_ndof = 0
        self.struct_nprocs = 0

        # Transfer data to keep track of
        self.xfer_ndof = 3
        self.meld = None

    def add_fsi_subsystems(self,model,step,prefix='',reuse_solvers=True):
        self.tacs_setup = setup['tacs']
        self.meld_setup = setup['meld']
        self.adflow_setup = setup['adflow']

        # Structural data to keep track of
        self.struct_nprocs = tacs_setup['nprocs']

        # Initialize the disciplinary meshes
        struct_mesh = self._initialize_meshes(reuse_solvers)

        # Initialize the disciplinary solvers
        aero, struct, disp_xfer, load_xfer = self._initialize_solvers()

        # Initialize the coupling group
        FsiSolver(aero=aero,struct=struct,disp_xfer=disp_xfer,load_xfer=load_xfer)

        # Initialize the function evaluators
        struct_funcs = self._initialize_function_evaluators()

        model.add_subsystem(prefix+'struct_mesh',struct_mesh,promotes['x_s0']

        model.add_subsystem(prefix+'FsiSolver',FsiSolver,promotes=['struct_dv','x_a0','x_s0','x_g','q','u_s'])

        model.add_subsystem(prefix+'struct_funcs',struct_funcs,promotes=['*']

    def _initialize_mesh(self,reuse_solvers):
        """
        Initialize the different discipline meshes
        """
        tacs_mesh   = TacsMesh(tacs_mesh_setup=self.tacs_mesh_setup)
        struct      = StructuralGroup(struct_comp=tacs_mesh,nprocs=self.struct_nprocs)

        return struct

    def _initialize_solvers(self):
        """
        Initialize the different disciplinary solvers
        """
        # Initialize the structural solver
        tacs   = TacsSolver(mesh_file=input_file,add_elements=add_elements)
        struct = StructuralGroup(struct_comp=tacs,nprocs=self.struct_nprocs)

        # Initialize the aero solver
        aero = ADFlow()

        # Initialize the transfers
        disp_xfer = FuntofemDisplacementTransfer()
        load_xfer = FuntofemLoadTransfer()

        return aero, struct, disp_xfer, load_xfer

    def _initialize_function_evaluators(self):
        # Set up the structural solver
        tacs_funcs = TacsFunctions(self.tacs_setup)
        struct_funcs = StructuralGroup(struct_comp=tacs_funcs,nprocs=self.struct_nprocs)
        return struct_funcs

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
        return self.tacs

    def tacs_solver_setup(self,comm):
        """
        Setup callback function for TACS solver setup.
        """

        mat = self.tacs.createFEMat()
        pc = TACS.Pc(mat)

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

        return func_list, self.tacs, self.struct_ndv

    def disp_xfer_setup(self,comm):
        isym = self.meld_setup['isym']
        n    = self.meld_setup['n']
        beta = self.meld_setup['beta']

        # Assumption: The transfer comm is the same as the flow comm
        self.meld = TransferScheme.pyMELD(comm,comm,0,self.struct_comm,0,isym,n,beta)
        return self.meld, self.xfer_ndof

    def load_xfer_setup(self,comm):
        return self.meld, self.xfer_ndof

    def adflow_setup(self,comm):

