from __future__ import division, print_function

from openmdao.api import NonlinearRunOnce, LinearRunOnce

from tacs_component import TacsMesh, TacsSolver, TacsFunctions
from tacs_component import PrescribedLoad

from tacs import TACS, functions

class TacsComps(object):
    """
    This class is design for adding steady TACS analysis to an OpenMDAO model/group.
    It contains some setup callback functions so that a tacs object can be shared
    between multiple OpenMDAO components

    """
    def __init__(self):
        # Structural data to keep track of
        self.tacs = None
        self.struct_ndv = 0
        self.struct_ndof = 0
        self.struct_nprocs = 0

    def add_tacs_subsystems(self,model,setup,prefix='',reuse_solvers=True,load_function=None):
        self.tacs_setup = setup['tacs']

        # Structural data to keep track of
        self.struct_nprocs = self.tacs_setup['nprocs']

        # Initialize the disciplinary mesh
        struct_mesh = self._initialize_mesh(reuse_solvers)

        # Initialize the disciplinary solver
        struct_solver = self._initialize_solver()
        model.nonlinear_solver = NonlinearRunOnce()
        model.linear_solver = LinearRunOnce()

        # Initialize the function evaluators
        struct_funcs = self._initialize_function_evaluator()

        model.add_subsystem(prefix+'struct_mesh',struct_mesh,promotes=['x_s0'],max_procs=self.struct_nprocs)

        if load_function is not None:
            struct_loads = PrescribedLoad(load_function=load_function,get_tacs=self.get_tacs)
            model.add_subsystem('struct_loads',struct_loads,promotes=['x_s0','f_s'],max_procs=self.struct_nprocs)

        model.add_subsystem(prefix+'struct_solver',struct_solver,promotes=['dv_struct','x_s0','u_s','f_s'],max_procs=self.struct_nprocs)
        model.add_subsystem(prefix+'struct_funcs',struct_funcs,promotes=['*'],max_procs=self.struct_nprocs)

    def _initialize_mesh(self,reuse_solvers):
        """
        Initialize the mesh
        """
        tacs_mesh   = TacsMesh(tacs_mesh_setup=self.tacs_mesh_setup)
        return tacs_mesh

    def _initialize_solver(self):
        """
        Initialize the TACS solver
        """
        tacs_solver   = TacsSolver(tacs_solver_setup=self.tacs_solver_setup)
        return tacs_solver

    def _initialize_function_evaluator(self):
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

    def get_tacs(self):
        return self.tacs
