# --- Python 3.8 ---
"""
@File    :   test_dafoam_derivs.py
@Time    :   2021/07/28
@Author  :   Bernardo Pacini
@Desc    :   File for testing the derivatives of the mphys dafoam wrapper
"""

# === Standard Python Modules ===
import unittest
import os

# === External Python modules ===
import shutil
import numpy as np

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint
from mphys.scenario_aerodynamic import ScenarioAerodynamic

baseDir = os.path.dirname(os.path.abspath(__file__))
dataDir = baseDir + "/../input_files/"

inputDirs = ["0", "constant", "FFD", "system"]

for dir in inputDirs:
    shutil.copytree(dataDir + dir, dir)


class Top(Multipoint):
    def setup(self):
        from mphys.mphys_dafoam import DAFoamBuilder
        from mphys.mphys_dvgeo import OM_DVGEOCOMP

        self.U0 = 10.0

        daOptions = {
            "designSurfaces": ["wing"],
            "solverName": "DASimpleFoam",
            "adjJacobianOption": "JacobianFree",
            "primalMinResTol": 1.0e-8,
            "primalBC": {
                "U0": {"variable": "U", "patches": ["inout"], "value": [self.U0, 0.0, 0.0]},
                "useWallFunction": True,
            },
            "objFunc": {
                "CD": {
                    "part1": {
                        "type": "force",
                        "source": "patchToFace",
                        "patches": ["wing"],
                        "directionMode": "parallelToFlow",
                        "alphaName": "alpha",
                        "scale": 1.0 / (0.5 * 10.0 * 10.0 * 45.5),
                        "addToAdjoint": True,
                    }
                },
                "CL": {
                    "part1": {
                        "type": "force",
                        "source": "patchToFace",
                        "patches": ["wing"],
                        "directionMode": "normalToFlow",
                        "alphaName": "alpha",
                        "scale": 1.0 / (0.5 * 10.0 * 10.0 * 45.5),
                        "addToAdjoint": True,
                    }
                },
            },
            "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm"},
            "normalizeStates": {
                "U": 10.0,
                "p": 100.0,
                "nuTilda": 1e-3,
                "phi": 1.0,
            },
            "adjPartDerivFDStep": {"State": 1e-6, "FFD": 1e-3},
            "checkMeshThreshold": {
                "maxAspectRatio": 1000.0,
                "maxNonOrth": 70.0,
                "maxSkewness": 8.0,
                "maxIncorrectlyOrientedFaces": 0,
            },
            "designVar": {
                "alpha": {"designVarType": "AOA", "patches": ["inout"], "flowAxis": "x", "normalAxis": "y"},
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

        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the mesh and cruise scenario because we only have one analysis point
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(ffd_file="FFD/wingFFD.xyz"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        self.cruise.coupling.mphys_set_dvgeo(self.geometry.DVGeo)

        # set the aero problem in the coupling and post coupling groups
        self.cruise.coupling.mphys_set_dvs_and_cons()
        self.cruise.aero_post.mphys_set_dvs_and_cons()

        self.cruise.aero_post.mphys_add_funcs(["CD", "CL"])

        # create geometric DV setup
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero", points)

        # add these points to the geometry object
        # self.geo.nom_add_point_dict(points)
        # create constraint DV setup
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup

        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="wingAxis", xFraction=0.25, alignIndex="k")

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z["wingAxis"].coef[i] = -val[i - 1]

        def alpha(val, DASolver):
            aoa = val[0]
            U = [float(self.U0 * np.cos(aoa)), float(self.U0 * np.sin(aoa)), 0]
            DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": U}})
            DASolver.updateDAOption()

        self.cruise.coupling.solver.aoa_func = alpha
        self.cruise.aero_post.aoa_func = alpha

        self.geometry.nom_addGeoDVGlobal(dvName="twist", value=np.array([0] * (nRefAxPts - 1)), func=twist)
        nShapes = self.geometry.nom_addGeoDVLocal(dvName="shape")

        leList = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
        teList = [[4.9, 0, 0.01], [8.9, 0, 13.9]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_add_LETEConstraint("lecon", 0, "iLow")
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh")

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([0] * (nRefAxPts - 1)))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa", val=np.array([0]))
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("aoa", "cruise.aoa")


class TestDAFoam(unittest.TestCase):
    def setUp(self):

        prob = om.Problem()
        prob.model = Top()

        # DVs
        # define the design variables
        prob.model.add_design_var("twist", lower=-10.0, upper=10.0, scaler=1.0)
        prob.model.add_design_var("shape", lower=-1.0, upper=1.0, scaler=1.0)
        prob.model.add_design_var("aoa", units="deg", lower=5.0, upper=10.0, scaler=1.0)

        # add constraints and the objective
        prob.model.add_objective("cruise.aero_post.CD", scaler=1.0)
        prob.model.add_constraint("cruise.aero_post.CL", equals=0.3, scaler=1.0)
        prob.model.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0)
        prob.model.add_constraint("geometry.volcon", lower=1.0, scaler=1.0)
        prob.model.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True)
        prob.model.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True)

        prob.setup()

        self.prob = prob
        # om.n2(
        #     prob,
        #     show_browser=False,
        #     outfile="test_dafoam_derivs.html" ,
        # )

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()

        data = self.prob.check_totals(wrt="twist", step=1e-3)
        for var, err in data.items():
            abs_err = err["abs error"]
            assert_near_equal(abs_err.forward, 0.0, 1e-3)

        data = self.prob.check_totals(wrt="shape", step=1e-3)
        for var, err in data.items():
            abs_err = err["abs error"]
            assert_near_equal(abs_err.forward, 0.0, 1e-3)

        data = self.prob.check_totals(wrt="aoa", step=1e-3)
        for var, err in data.items():
            abs_err = err["abs error"]
            assert_near_equal(abs_err.forward, 0.0, 1e-3)


if __name__ == "__main__":
    unittest.main()
