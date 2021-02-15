# --- Python 3.8 ---
"""
@File    :   test_geom_derivs.py
@Time    :   2020/12/19
@Author  :   Josh Anibal
@Desc    :   Test the dervatives of the geometric components
"""

# === Standard Python modules ===
import unittest
import os

# === External Python modules ===
import numpy as np

# from mpi4py import MPI

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint
from adflow import ADFLOW

from mphys.mphys_dvgeo import OM_DVGEOCOMP


baseDir = os.path.dirname(os.path.abspath(__file__))


class Top(om.Group):
    def setup(self):

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc for DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        # geometry parametrization with FFD and general geometric constraints
        self.add_subsystem(
            "geo",
            OM_DVGEOCOMP(
                ffd_file=os.path.join(baseDir, "../input_files/ffd.xyz"),
            ),
        )

    def configure(self):

        # ################################################################################
        # # GEOMETRY SETUP
        # ################################################################################

        points = {}

        aero_options = {
            # I/O Parameters
            "gridFile": os.path.join(baseDir, "../input_files/wing_vol_L3.cgns"),
            "outputDirectory": os.path.join(baseDir, "../output_files"),
            "monitorvariables": ["resrho", "resturb", "cl", "cd"],
            "surfacevariables": ["cp", "vx", "vy", "vz", "mach"],
            # 'isovariables': ['shock'],
            "isoSurface": {"shock": 1},  # ,'vx':-0.0001},
            "writeTecplotSurfaceSolution": False,
            "writevolumesolution": False,
            'writesurfacesolution':False,
            "liftindex": 3,
            # Physics Parameters
            "equationType": "RANS",
            # Solver Parameters
            "smoother": "DADI",
            "CFL": 1.5,
            "MGCycle": "sg",
            "MGStartLevel": -1,
            # ANK Solver Parameters
            "useANKSolver": True,
            "nsubiterturb": 5,
            "anksecondordswitchtol": 1e-4,
            "ankcoupledswitchtol": 1e-6,
            "ankinnerpreconits": 2,
            "ankouterpreconits": 1,
            "anklinresmax": 0.1,
            "infchangecorrection": True,
            "ankcfllimit": 1e4,
            # NK Solver Parameters
            # 'useNKSolver':True,
            "nkswitchtol": 1e-4,
            # Termination Criteria
            "L2Convergence": 1e-15,
            "L2ConvergenceCoarse": 1e-2,
            "L2ConvergenceRel": 1e-14,
            "nCycles": 10000,
            "adjointl2convergencerel": 1e-14,
            "adjointl2convergence": 1e-14
            # force integration
        }

        CFDSolver = ADFLOW(options=aero_options)

        points["aero_points"] = CFDSolver.getSurfaceCoordinates(includeZipper=False).flatten(order="C")

        # add these points to the geometry object
        self.geo.nom_add_point_dict(points)
        # create constraint DV setup
        # tri_points = self.mp.mphys_get_triangulated_surface()
        # self.geo.nom_setConstraintSurface(tri_points)

        # # connect
        # for key in points:
        #     self.connect("geo.%s" % key, "mp.%s" % key)

        # add geometric DVs
        nRefAxPts = self.geo.nom_addRefAxis(name="wing", xFraction=0.25, alignIndex="k")
        nTwist = nRefAxPts - 1

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_y["wing"].coef[i] = val[i - 1]

        self.geo.nom_addGeoDVGlobal(dvName="twist", value=np.zeros(nTwist), func=twist)
        self.dvs.add_output("twist", val=np.array([0] * nTwist))
        self.connect("twist", "geo.twist")

        # shape DVs
        n_dv = self.geo.nom_addGeoDVLocal(dvName="local", axis="z")
        self.dvs.add_output("local", val=np.array([0] * n_dv))
        self.connect("local", "geo.local")


class TestDVGeo(unittest.TestCase):
    def setUp(self):

        ################################################################################
        # OpenMDAO setup
        ################################################################################
        prob = om.Problem()
        prob.model = Top()

        prob.model.add_design_var("twist")
        prob.model.add_design_var("local")

        # objectives and nonlinear constraints
        prob.model.add_constraint("geo.aero_points")

        prob.setup(mode="rev")
        # om.n2(
        #     prob,
        #     show_browser=False,
        #     outfile="test_geom_derivs.html",
        # )

        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        print("----------------strating check totals--------------")
        # data = self.prob.check_totals(step=1e-7, form="forward")
        data = self.prob.check_totals(step=1e-7)  # out_stream=None
        for var, err in data.items():

            rel_err = err["rel error"]  # ,  'rel error']
            assert_near_equal(rel_err.forward, 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()
