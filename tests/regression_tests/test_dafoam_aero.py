# --- Python 3.8 ---
"""
@File    :   test_dafoam_aero.py
@Time    :   2021/07/28
@Author  :   Bernardo Pacini
@Desc    :   Aerodynamic analysis regression test used to test if the output
             produced by MPHYS has changed
"""

import os
# === External Python modules ===
import shutil
# === Standard Python Modules ===
import unittest

import numpy as np
# === Extension modules ===
import openmdao.api as om
from mpi4py import MPI
from openmdao.utils.assert_utils import assert_near_equal
from parameterized import parameterized, parameterized_class

from mphys.multipoint import Multipoint
from mphys.scenarios.aerodynamic import ScenarioAerodynamic

baseDir = os.path.dirname(os.path.abspath(__file__))
dataDir = baseDir + "/../input_files/dafoam_aero/"

inputDirs = ["0", "constant", "FFD", "system"]

for dir in inputDirs:
    shutil.copytree(dataDir + dir, dir)


class Top(Multipoint):
    def setup(self):
        from dafoam.mphys import DAFoamBuilder
        from pygeo.mphys import OM_DVGEOCOMP

        self.U0 = 10.0

        daOptions = {
            "designSurfaces": ["wing"],
            "solverName": "DASimpleFoam",
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
                        "alphaName": "aoa",
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
                        "alphaName": "aoa",
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
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=dafoam_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        super().configure()

        self.cruise.aero_post.mphys_add_funcs()

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

        def aoa(val, DASolver):
            aoa = val[0] * np.pi / 180.0
            U = [float(self.U0 * np.cos(aoa)), float(self.U0 * np.sin(aoa)), 0]
            DASolver.setOption("primalBC", {"U0": {"variable": "U", "patches": ["inout"], "value": U}})
            DASolver.updateDAOption()

        self.cruise.coupling.solver.add_dv_func("aoa", aoa)
        self.cruise.aero_post.add_dv_func("aoa", aoa)

        self.geometry.nom_addGlobalDV(dvName="twist", value=np.array([0] * (nRefAxPts - 1)), func=twist)
        nShapes = self.geometry.nom_addLocalDV(dvName="shape")

        leList = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
        teList = [[4.9, 0, 0.01], [8.9, 0, 13.9]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_add_LETEConstraint("lecon", 0, "iLow")
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh")

        # add dvs to ivc and connect
        self.dvs.add_output("twist", val=np.array([0] * (nRefAxPts - 1)))
        self.dvs.add_output("shape", val=np.array([0] * nShapes))
        self.dvs.add_output("aoa", units="deg", val=np.array([0]))
        self.connect("twist", "geometry.twist")
        self.connect("shape", "geometry.shape")
        self.connect("aoa", "cruise.aoa")


@parameterized_class(
    [
        {
            "ref_vals": {"CD": 0.03183567874922066, "CL": 0.13061624040077277},
        },
    ]
)
class TestDAFoam(unittest.TestCase):
    def setUp(self):

        prob = om.Problem()
        prob.model = Top()

        prob.setup()

        self.prob = prob
        # om.n2(
        #     prob,
        #     show_browser=False,
        #     outfile="test_dafoam_derivs.html" ,
        # )

    def test_run_model(self):
        self.prob.run_model()
        # self.prob.model.list_outputs()
        if MPI.COMM_WORLD.rank == 0:
            assert_near_equal(self.prob.get_val("cruise.aero_post.CL"), self.ref_vals["CL"], 1e-6)

            assert_near_equal(self.prob.get_val("cruise.aero_post.CD"), self.ref_vals["CD"], 1e-6)


if __name__ == "__main__":
    unittest.main()
