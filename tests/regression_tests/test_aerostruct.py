# --- Python 3.8 ---
"""
@File    :   test_aerostruct.py
@Time    :   2020/12/20
@Author  :   Josh Anibal
@Desc    :   Aerostructural regression tests used to test if the output produced
             by MPHYS has changed
"""

# === Standard Python modules ===
from __future__ import print_function, division
import os
import unittest


# === External Python modules ===
import numpy as np
from mpi4py import MPI
from parameterized import parameterized, parameterized_class


# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


from mphys.multipoint import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural

# these imports will be from the respective codes' repos rather than mphys
from adflow.mphys import ADflowBuilder
from tacs.mphys import TacsBuilder
from funtofem.mphys import MeldBuilder
# TODO RLT needs to be updated with the new tacs wrapper
# from rlt.mphys import RltBuilder

from baseclasses import AeroProblem
from tacs import elements, constitutive, functions


# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank


baseDir = os.path.dirname(os.path.abspath(__file__))

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    rho = 2780.0  # density, kg/m^3
    E = 73.1e9  # elastic modulus, Pa
    nu = 0.33  # poisson's ratio
    ys = 324.0e6  # yield stress, Pa
    thickness = 0.003
    min_thickness = 0.002
    max_thickness = 0.05

    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=thickness, tNum=dvNum, tlb=min_thickness, tub=max_thickness)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None
    elem = elements.Quad4Shell(transform, con)

    return elem

def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Add TACS Functions
    # Only include mass from elements that belong to pytacs components (i.e. skip concentrated masses)
    problem.addFunction('mass', functions.StructuralMass)
    problem.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

    # Add gravity load
    g = np.array([0.0, 0.0, -9.81])  # m/s^2
    problem.addInertialLoad(g)


class Top(Multipoint):
    def setup(self):

        ################################################################################
        # ADflow options
        ################################################################################
        aero_options = {
            # I/O Parameters
            "gridFile": os.path.join(baseDir, "../input_files/wing_vol_L3.cgns"),
            "outputDirectory": ".",
            "monitorvariables": ["resrho", "resturb", "cl", "cd"],
            "writeTecplotSurfaceSolution": False,
            # 'writevolumesolution':False,
            # 'writesurfacesolution':False,
            # Physics Parameters
            "equationType": "RANS",
            "liftindex": 3,
            # Solver Parameters
            "smoother": "DADI",
            "CFL": 1.5,
            "CFLCoarse": 1.25,
            "MGCycle": "sg",
            "MGStartLevel": -1,
            "nCyclesCoarse": 250,
            # ANK Solver Parameters
            "useANKSolver": True,
            "nsubiterturb": 5,
            "anksecondordswitchtol": 1e-4,
            "ankcoupledswitchtol": 1e-6,
            "ankinnerpreconits": 2,
            "ankouterpreconits": 2,
            "anklinresmax": 0.1,
            # Termination Criteria
            "L2Convergence": 1e-14,
            "L2ConvergenceCoarse": 1e-2,
            "L2ConvergenceRel": 1e-4,
            "nCycles": 10000,
            # force integration
            "forcesAsTractions": self.forcesAsTractions,
        }

        aero_builder = ADflowBuilder(aero_options, scenario="aerostructural")
        aero_builder.initialize(self.comm)
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # TACS setup
        ################################################################################
        struct_builder = TacsBuilder(mesh_file='../input_files/wingbox.bdf', element_callback=element_callback,
                                     problem_setup=problem_setup, coupled=True)
        struct_builder.initialize(self.comm)

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Transfer scheme options
        ################################################################################
        if self.xfer_builder_class == MeldBuilder:
            xfer_builder = self.xfer_builder_class(aero_builder, struct_builder, isym=1, check_partials=True)
        else:
            xfer_builder = self.xfer_builder_class(self.xfer_options, aero_builder, struct_builder, check_partials=True)
        xfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        self.mphys_add_scenario(
            "cruise",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["aero", "struct"]:
            self.mphys_connect_scenario_coordinate_source("mesh_%s" % discipline, "cruise", discipline)

        # add the structural thickness DVs
        ndv_struct = struct_builder.get_ndv()
        dvs.add_output("dv_struct", np.array(ndv_struct * [0.01]))
        self.connect("dv_struct", "cruise.dv_struct")

    def configure(self):
        super().configure()

        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        aoa = 1.5
        ap0 = AeroProblem(
            name="ap0",
            mach=0.8,
            altitude=10000,
            alpha=aoa,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=["lift", "drag", "cl", "cd"],
        )
        ap0.addDV("alpha", value=aoa, name="aoa")
        ap0.addDV("mach", value=0.8, name="mach")

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.cruise.coupling.aero.mphys_set_ap(ap0)
        self.cruise.aero_post.mphys_set_ap(ap0)

        # define the aero DVs in the IVC
        # s0
        self.dvs.add_output("aoa0", val=aoa, units="deg")
        self.dvs.add_output("mach0", val=0.8)

        # connect to the aero for each scenario
        self.connect("aoa0", ["cruise.coupling.aero.aoa", "cruise.aero_post.aoa"])
        self.connect("mach0", ["cruise.coupling.aero.mach", "cruise.aero_post.mach"])


@parameterized_class(
    [
        {
            "name": "meld",
            "xfer_builder_class": MeldBuilder,
            "xfer_options": {"isym": 1, "n": 200, "beta": 0.5},
            "ref_vals": {
                "xa": 5.633698956781166,
                "cl": 0.209180113189859,
                "func_struct": 0.6808193125485819,
                "cd": 0.025108879963099927,
            },
        },
        # {
        #     "name": "rlt",
        #     # "xfer_builder_class": RltBuilder,
        #     "xfer_options": {"transfergaussorder": 2},
        #     "ref_vals": {"xa": 5.504999831790868, "func_struct": 0.31363742, "cl": 0.3047756, "cd": 0.0280476},
        # },
    ]
)
class TestAeroStructSolve(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):

        ################################################################################
        # OpenMDAO setup
        ################################################################################

        prob = om.Problem()
        prob.model = Top()

        prob.model.xfer_builder_class = self.xfer_builder_class
        prob.model.xfer_options = self.xfer_options

        if "meld" in self.name:
            prob.model.forcesAsTractions = True
        else:
            prob.model.forcesAsTractions = False

        prob.setup()

        self.prob = prob
        # om.n2(prob, show_browser=False, outfile='test_as.html')

    def test_run_model(self):
        self.prob.run_model()
        # prob.model.list_outputs()
        if MPI.COMM_WORLD.rank == 0:
            print("Scenario 0")

            print("xa =", np.mean(self.prob.get_val("cruise.coupling.geo_disp.x_aero", get_remote=True)))
            print("cl =", self.prob.get_val("cruise.aero_post.cl", get_remote=True)[0])
            print("cd =", self.prob.get_val("cruise.aero_post.cd", get_remote=True)[0])
            print("ks_vmfailure =", self.prob.get_val("cruise.ks_vmfailure", get_remote=True)[0])

            assert_near_equal(
                np.mean(self.prob.get_val("cruise.coupling.geo_disp.x_aero", get_remote=True)),
                self.ref_vals["xa"],
                1e-6,
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.aero_post.cl", get_remote=True)), self.ref_vals["cl"], 1e-6
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.aero_post.cd", get_remote=True)), self.ref_vals["cd"], 1e-6
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.ks_vmfailure", get_remote=True)),
                self.ref_vals["func_struct"],
                1e-6,
            )



if __name__ == "__main__":
    unittest.main()
