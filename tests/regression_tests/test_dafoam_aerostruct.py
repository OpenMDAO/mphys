# --- Python 3.8 ---
"""
@File    :   test_dafoam_aerostruct.py
@Time    :   2021/12/17
@Author  :   Bernardo Pacini
@Desc    :   DAFoam aerostructural regression tests used to test if the output
             produced by MPHYS has changed
"""
# === Standard Python modules ===
import os
import unittest

# === External Python modules ===
import shutil
import numpy as np
from mpi4py import MPI
from parameterized import parameterized, parameterized_class

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural

# these imports will be from the respective codes' repos rather than mphys
from dafoam.mphys import DAFoamBuilder
from tacs.mphys import TacsBuilder
from pyfuntofem import MeldBuilder
from pygeo.mphys import OM_DVGEOCOMP

from tacs import elements, constitutive, functions

baseDir = os.path.dirname(os.path.abspath(__file__))
dataDir = baseDir + "/../input_files/dafoam_aerostruct/"

inputDirs = ["0", "constant", "FFD", "system"]
inputFiles = ["wingbox.bdf"]

for dir in inputDirs:
    shutil.copytree(dataDir + dir, dir)

for fil in inputFiles:
    shutil.copyfile(dataDir + fil, fil)

U0 = 200.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
CL_target = 0.3
aoa0 = 2.0
rho0 = p0 / T0 / 287.0
A0 = 45.5


class Top(Multipoint):
    def setup(self):
        daOptions = {
            "designSurfaces": ["wing"],
            "solverName": "DARhoSimpleFoam",
            "fsi": {
                "pRef": p0,
            },
            "adjUseColoring": False,
            "primalMinResTol": 1.0e-8,
            "debug": False,
            "primalBC": {
                "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
                "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
                "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
                "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
                "useWallFunction": False,
            },
            # variable bounds for compressible flow conditions
            "primalVarBounds": {
                "UMax": 1000.0,
                "UMin": -1000.0,
                "pMax": 500000.0,
                "pMin": 20000.0,
                "eMax": 500000.0,
                "eMin": 100000.0,
                "rhoMax": 5.0,
                "rhoMin": 0.2,
            },
            "objFunc": {
                "CD": {
                    "part1": {
                        "type": "force",
                        "source": "patchToFace",
                        "patches": ["wing"],
                        "directionMode": "parallelToFlow",
                        "alphaName": "aoa",
                        "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                        "addToAdjoint": True,
                    }
                },
                "CL": {
                    "part1": {
                        "type": "force",
                        "source": "patchToFace",
                        "patches": ["wing"],
                        "directionMode": "normalToFlow",
                        "alphaName": "aoa",
                        "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
                        "addToAdjoint": True,
                    }
                },
            },
            "adjEqnOption": {
                "gmresRelTol": 1.0e-2,
                "pcFillLevel": 1,
                "jacMatReOrdering": "rcm",
                "useNonZeroInitGuess": True,
            },
            "normalizeStates": {
                "U": U0,
                "p": p0,
                "T": T0,
                "nuTilda": 1e-3,
                "phi": 1.0,
            },
            "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
            "checkMeshThreshold": {
                "maxAspectRatio": 5000.0,
                "maxNonOrth": 70.0,
                "maxSkewness": 8.0,
                "maxIncorrectlyOrientedFaces": 0,
            },
            "designVar": {
                "aoa": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
                "twist": {"designVarType": "FFD"},
                "shape": {"designVarType": "FFD"},
            },
        }

        meshOptions = {
            "gridFile": os.getcwd(),
            "fileType": "OpenFOAM",
            "useRotations": False,
            # point and normal for the symmetry plane
            "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        }

        aero_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerostructural")
        aero_builder.initialize(self.comm)
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # TACS options
        ################################################################################
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
            problem.addFunction("mass", functions.StructuralMass)
            problem.addFunction("ks_vmfailure", functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

            # Add gravity load
            g = np.array([0.0, 0.0, -9.81])  # m/s^2
            problem.addInertialLoad(g)

        # TACS Setup
        tacs_options = {
            "element_callback": element_callback,
            "problem_setup": problem_setup,
            "mesh_file": "wingbox.bdf",
        }

        struct_builder = TacsBuilder(tacs_options)
        struct_builder.initialize(self.comm)

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Transfer scheme options
        ################################################################################
        xfer_builder = MeldBuilder(aero_builder, struct_builder, isym=2, check_partials=True)
        xfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-8)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-8)
        self.mphys_add_scenario(
            "cruise",
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=xfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["aero"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0_masked" % discipline)
        for discipline in ["struct"]:
            self.connect("geometry.x_%s0" % discipline, "cruise.x_%s0" % discipline)
        # add the structural thickness DVs
        ndv_struct = struct_builder.get_ndv()
        dvs.add_output("dv_struct", np.array(ndv_struct * [0.01]))
        self.connect("dv_struct", "cruise.dv_struct")

        self.connect("mesh_aero.x_aero0", "geometry.x_aero_in")
        self.connect("mesh_struct.x_struct0", "geometry.x_struct_in")

    def configure(self):
        super().configure()

        self.cruise.aero_post.mphys_add_funcs()

        # create geometric DV setup
        points = self.mesh_aero.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)
        self.geometry.nom_add_discipline_coords("struct")

        # create constraint DV setup
        tri_points = self.mesh_aero.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(U0 * np.cos(aoa)), float(U0 * np.sin(aoa)), 0]
            DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": U}})
            DASolver.updateDAOption()

        self.cruise.coupling.aero.solver.add_dv_func("aoa", aoa)
        self.cruise.aero_post.add_dv_func("aoa", aoa)

        # add dvs to ivc and connect
        self.dvs.add_output("aoa", val=np.array([aoa0]))
        self.connect("aoa", "cruise.aoa")

        # define the design variables
        self.add_design_var("aoa", lower=0.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        self.add_objective("cruise.aero_post.CD", scaler=1.0)
        self.add_constraint("cruise.aero_post.CL", equals=CL_target, scaler=1.0)


@parameterized_class(
    [
        {
            "name": "meld",
            "ref_vals": {
                "xa": 5.352438021381199,
                "cl": 0.3196186044913372,
                "cd": 0.0121836189015169,
                "mass": 2165.4853211276463,
                "ks_vmfailure": 0.5578983727565848,
            },
        },
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

        prob.setup()

        self.prob = prob
        # om.n2(prob, show_browser=False, outfile='test_as.html')

    def test_run_model(self):
        self.prob.run_model()
        # prob.model.list_outputs()
        if MPI.COMM_WORLD.rank == 0:
            print("Scenario 0")

            print("xa =", np.mean(self.prob.get_val("cruise.coupling.geo_disp.x_aero", get_remote=True)))
            print("cl =", self.prob.get_val("cruise.aero_post.CL", get_remote=True))
            print("cd =", self.prob.get_val("cruise.aero_post.CD", get_remote=True))
            print("mass =", self.prob.get_val("cruise.mass", get_remote=True))
            print("ks_vmfailure =", self.prob.get_val("cruise.ks_vmfailure", get_remote=True))

            assert_near_equal(
                np.mean(self.prob.get_val("cruise.coupling.geo_disp.x_aero", get_remote=True)),
                self.ref_vals["xa"],
                1e-6,
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.aero_post.CL", get_remote=True)), self.ref_vals["cl"], 1e-6
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.aero_post.CD", get_remote=True)), self.ref_vals["cd"], 1e-6
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.mass", get_remote=True)),
                self.ref_vals["mass"],
                1e-6,
            )
            assert_near_equal(
                np.mean(self.prob.get_val("cruise.ks_vmfailure", get_remote=True)),
                self.ref_vals["ks_vmfailure"],
                1e-6,
            )


if __name__ == "__main__":
    unittest.main()
