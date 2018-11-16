from fsi_group import FsiSolver

from tacs_component import TacsSolver, StructuralGroup, TacsFunctions
from aero_group import AeroSolver
from displacement_xfer_component import FuntofemDisplacementTransfer
from load_xfer_component import FuntofemLoadTransfer

def setup_elements(mesh):
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
    return ndof, ndv

def setup_tacs_functions():

class FsiComps(object):
    """
    This class contains some setup callback functions so that TACS, ADFlow, and
    MELD objects can be shared between OpenMDAO components
    """
    def __init__(self,tacs_setup,meld_setup,adflow_setup):
        self.tacs_setup = tacs_setup
        self.meld_setup = meld_setup
        self.adflow_setup = adflow_setup

    def setup_solvers(self,comm):
        # Set up the structural solver
        tacs   = TacsSolver(mesh_file=input_file,add_elements=add_elements)
        struct = StructuralGroup(struct_comp=tacs,nprocs=4)

        # Set up the aero solver
        aero = ADFlow()

        # Set up the transfers
        disp_xfer = FuntofemDisplacementTransfer()
        load_xfer = FuntofemLoadTransfer()

        return aero, struct, disp_xfer, load_xfer

    def tacs_setup(self,comm):
        """
        Setup callback function for TACS.
        """
        mesh_file = self.tacs_setup['mesh_file']

        mesh = TACS.MeshLoader(comm)
        mesh.scanBDFFile(mesh_file)

        self.struct_ndof, self.struct_ndv = add_elements(mesh)
       
        self.tacs = mesh.createTACS(self.struct_ndof)

        mat = tacs.createFEMat()
        pc = TACS.Pc(mat)

        nrestart = 0 # number of restarts before giving up
        m = 30 # size of Krylov subspace (max # of iterations)
        gmres = TACS.KSM(mat, pc, m, nrestart)

        return self.tacs, pc, gmres, self.ndv

    def tacs_func_setup(self,comm):

        return self.tacs, func_list, self.ndv

    def disp_xfer_setup(self,comm):
        isym = self.meld_setup['isym']
        n    = self.meld_setup['n']
        beta = self.meld_setup['beta']

        struct_ndof = 3

        self.meld = TransferScheme.pyMELD(comm,comm,0,comm,0,isym,n,beta)
        return self.meld, self.struct_ndof

    def load_xfer_setup(self,comm):
        return self.meld, self.xfer_ndof

    def adflow_setup(self,comm):

    def setup_function_evaluators(self,comm):
        # Set up the structural solver
        tacs = TacsSolver(mesh_file='input.bdf',add_elements=add_elements)
        struct = StructuralGroup(struct_comp=tacs,nprocs=4)

        # Set up the aero solver
        aero = ADFlow()

        # Set up the transfers
        disp_xfer = FuntofemDisplacementTransfer()
        load_xfer = FuntofemLoadTransfer(meld=disp_xfer.meld)

        return aero, struct, disp_xfer, load_xfer

def create_model():

    FsiSolver(setup=setup_func)

    # Set up the geometry
    geom = Geometry()

    # Add function evaluations
    tacs_funcs = 

    # Put it all together
    prob = Problem()
    model = prob.model

    model.add_subsystem('FSI',FsiSolver,promotes=['struct_dv','x_a0','x_s0','x_g','q','u_s'])
    model.add_subsystem('geom',geom,promotes=['shape_dv','x_a0','x_s0'])
