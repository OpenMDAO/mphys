
from omfsi.fsi_group import FsiSolver
from omfsi.tacs_component import TacsMesh, TacsSolver, TacsFunctions
from omfsi.displacement_xfer_component import FuntofemDisplacementTransfer
from omfsi.load_xfer_component import FuntofemLoadTransfer
from omfsi.aero_groups import AeroSolverGroup

from dummy_aero_fixed import AeroMesh, AeroDeformer, AeroSolver, AeroForceIntegrator

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
    def __init__(self):
        # Flow data to keep track of
        self.flow = None

        # Structural data to keep track of
        self.struct_comm = None
        self.tacs = None
        self.struct_ndv = 0
        self.struct_ndof = 0
        self.struct_nprocs = 0

        # Transfer data to keep track of
        self.xfer_ndof = 3
        self.meld = None

    def add_fsi_subsystems(self,model,setup,prefix='',reuse_solvers=True):
        self.tacs_setup = setup['tacs']
        self.meld_setup = setup['meld']
        self.flow_setup = setup['flow']

        # Structural data to keep track of
        self.struct_nprocs = self.tacs_setup['nprocs']

        # Initialize the disciplinary meshes
        aero_mesh, struct_mesh = self._initialize_meshes(reuse_solvers)

        # Initialize the disciplinary solvers
        aero, struct, disp_xfer, load_xfer = self._initialize_solvers()

        # Initialize the coupling group
        fsi_solver = FsiSolver(aero=aero,
                               struct=struct,
                               disp_xfer=disp_xfer,
                               load_xfer=load_xfer,
                               struct_nprocs=self.struct_nprocs,
                               get_vector_size=self.get_vector_size)

        # Initialize the function evaluators
        struct_funcs = self._initialize_function_evaluators()

        model.add_subsystem(prefix+'aero_mesh',aero_mesh,promotes=['x_a0'])
        model.add_subsystem(prefix+'struct_mesh',struct_mesh,promotes=['x_s0'],max_procs=self.struct_nprocs)

        model.add_subsystem(prefix+'fsi_solver',fsi_solver,promotes=['dv_aero','dv_struct','x_a0','x_s0','x_g','q','u_s'])

        model.add_subsystem(prefix+'struct_funcs',struct_funcs,promotes=['*'],max_procs=self.struct_nprocs)

    def _initialize_meshes(self,reuse_solvers):
        """
        Initialize the disciplinary meshes
        """
        aero_mesh   = AeroMesh(aero_mesh_setup=self.aero_mesh_setup)
        tacs_mesh   = TacsMesh(tacs_mesh_setup=self.tacs_mesh_setup)

        return aero_mesh, tacs_mesh

    def _initialize_solvers(self):
        """
        Initialize the disciplinary solvers
        """
        deformer    = AeroDeformer(aero_deformer_setup=self.aero_deformer_setup)
        aero_solver = AeroSolver(aero_solver_setup=self.aero_solver_setup)
        aero_force  = AeroForceIntegrator(aero_force_integrator_setup=self.aero_force_integrator_setup)
        aero_group  = AeroSolverGroup(deformer=deformer,solver=aero_solver,force=aero_force)

        tacs_solver = TacsSolver(tacs_solver_setup=self.tacs_solver_setup)

        disp_xfer   = FuntofemDisplacementTransfer(disp_xfer_setup=self.disp_xfer_setup)
        load_xfer   =         FuntofemLoadTransfer(load_xfer_setup=self.load_xfer_setup)

        return aero_group, tacs_solver, disp_xfer, load_xfer

    def _initialize_function_evaluators(self):
        """
        Initialize the TACS function evaluator
        """
        tacs_funcs = TacsFunctions(tacs_func_setup=self.tacs_func_setup)
        return tacs_funcs

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

    def aero_mesh_setup(self,comm):
        """
        Setup callback function for TACS intial setup: reading the mesh and
        assigning elements
        """
        self.flow = {}
        self.flow['nnodes'] = 2
        self.aero_nnodes = self.flow['nnodes']
        return self.flow

    def aero_deformer_setup(self,comm):
        """
        Setup callback function for TACS solver setup.
        """

        return self.flow

    def aero_solver_setup(self,comm):
        """
        Setup callback function for TACS solver setup.
        """

        return self.flow

    def aero_force_integrator_setup(self,comm):
        """
        Setup callback function for TACS solver setup.
        """

        return self.flow

    def aero_func_setup(self,comm):
        """
        Setup callback function for TACS function evaluation.
        The user provides a list TACS function names (strings)
        """
        return self.flow
    def get_vector_size(self):
        return 3 * self.aero_nnodes

    def disp_xfer_setup(self,comm):
        isym = self.meld_setup['isym']
        n    = self.meld_setup['n']
        beta = self.meld_setup['beta']

        # Assumption: The transfer comm is the same as the aero comm
        self.meld = TransferScheme.pyMELD(comm,comm,0,self.struct_comm,0,isym,n,beta)
        return self.meld, self.aero_nnodes, self.struct_nnodes, self.struct_ndof

    def load_xfer_setup(self):
        return self.meld, self.aero_nnodes, self.struct_nnodes, self.struct_ndof
