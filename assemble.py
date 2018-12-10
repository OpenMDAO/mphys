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

    """
    def __init__(self):
        # Flow data to keep track of
        self.adflow = None

        # Structural data to keep track of
        self.struct_comm = None
        self.tacs = None
        self.struct_ndv = 0
        self.struct_ndof = 0

        # Transfer data to keep track of
        self.xfer_ndof = 3
        self.meld = None

    def add_fsi_subsystems(self,model,step,prefix='',reuse_solvers=True):
        self.tacs_setup = setup['tacs']
        self.meld_setup = setup['meld']
        self.adflow_setup = setup['adflow']

        # Structural data to keep track of
        self.struct_nprocs = tacs_setup['nprocs']

        # Initialize the disciplinary solvers
        aero, struct, disp_xfer, load_xfer = self._initialize_solvers(reuse_solvers)

        # Initialize the coupling group
        FsiSolver(aero=aero,struct=struct,disp_xfer=disp_xfer,load_xfer=load_xfer)

        # Initialize the function evaluators
        struct_funcs = self._initialize_function_evaluators()

        model.add_subsystem(prefix+'FsiSolver',FsiSolver,promotes=['struct_dv','x_a0','x_s0','x_g','q','u_s'])
        model.add_subsystem(prefix+'struct_funcs',struct_funcs,promotes=['*']

    def _initialize_solvers(self,reuse_solvers):
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

    def tacs_solver_setup(self,comm):
        """
        Setup callback function for TACS solver setup.
        The user provides the mesh file name and a function to add the elements
        to the mesh object.
        """
        self.struct_comm = comm
        mesh_file = self.tacs_setup['mesh_file']
        add_elements = self.tacs_setup['add_elements']

        mesh = TACS.MeshLoader(comm)
        mesh.scanBDFFile(mesh_file)

        self.struct_ndof, self.struct_ndv = add_elements(mesh)
       
        self.tacs = mesh.createTACS(self.struct_ndof)

        mat = tacs.createFEMat()
        pc = TACS.Pc(mat)

        nrestart = 0 # number of restarts before giving up
        m = 30 # size of Krylov subspace (max # of iterations)
        gmres = TACS.KSM(mat, pc, m, nrestart)

        return self.tacs, pc, gmres, self.struct_ndv

    def tacs_func_setup(self,comm):
        """
        Setup callback function for TACS function evaluation.
        The user provides a list TACS functions
        """
        func_list = self.tacs_setup['func_list']

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




# problem setup

def add_elements(mesh):
    rho = 2500.0  # density, kg/m^3
    E = 70.0e9 # elastic modulus, Pa
    nu = 0.3 # poisson's ratio
    kcorr = 5.0 / 6.0 # shear correction factor
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.001
    max_thickness = 0.100

    num_components = mesh
    for i in xrange(num_components)
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, t, i,
                                     min_thickness, max_thickness)
        element = None
        if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.MITCShell(2,stiff,component_num=i)
        mesh.setElement(i, element)

    ndof = 6
    ndv = num_components

    return ndof, ndv


func_list = ['ksfailure']
meld_setup = {'isym':1, 'n':200,'beta':0.5}
adflow_setup = {}
tacs_setup = {'add_elements': add_elements, 
              'nprocs'      : 4,
              'mesh_file'   : 'CRM_box_2nd.bdf'}
              'func_list'   : func_list}
