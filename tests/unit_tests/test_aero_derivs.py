# --- Python 3.8 ---
"""
@File    :   test_aero_derivs.py
@Time    :   2020/12/17
@Author  :   Josh Anibal
@Desc    :   File for testing the derivatives of the mphys adflow wrapper
"""

# === Standard Python modules ===
import unittest
import os

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint

from adflow import ADFLOW
from baseclasses import AeroProblem


baseDir = os.path.dirname(os.path.abspath(__file__))


class Top(om.Group):
    def setup(self):

        ################################################################################
        # AERO
        ################################################################################

        from mphys.mphys_adflow import ADflowBuilder

        aero_options = {
            # I/O Parameters
            "gridFile": os.path.join(baseDir, "../input_files/wing_vol_L3.cgns"),
            "outputDirectory": os.path.join(baseDir, "../output_files/"),
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

        aero_builder = ADflowBuilder(aero_options)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc for DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        CFDSolver = ADFLOW(options=aero_options)

        x_a = CFDSolver.getSurfaceCoordinates(groupName="allWalls")
        dvs.add_output("x_a", val=x_a)

        # create the multiphysics multipoint group.
        mp = self.add_subsystem("mp", Multipoint(aero_builder=aero_builder, struct_builder=None, xfer_builder=None))

        mp.mphys_add_scenario("s0")

    def configure(self):
        alpha0 = 3.725

        # configure aero DVs
        ap0 = AeroProblem(
            name="ap0",
            mach=0.85,
            altitude=10000,
            alpha=alpha0,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=["lift", "drag", "cl", "cd"],
        )
        ap0.addDV("alpha", value=2.0, name="alpha")
        self.mp.s0.solver_group.aero.mphys_set_ap(ap0)
        # self.mp.s0.solver_group.aero_funcs.mphys_set_ap(ap0)
        self.mp.s0.aero_funcs.mphys_set_ap(ap0)

        self.dvs.add_output("alpha", val=alpha0)
        self.connect("x_a", ["mp.aero_mesh.x_a0_points"])
        # self.connect("alpha", ["mp.s0.solver_group.aero.alpha", "mp.s0.aero_funcs.alpha"])

        self.mp.aero_mesh.mphys_add_coordinate_input()
        self.connect("alpha", ["mp.s0.solver_group.aero.alpha", "mp.s0.aero_funcs.alpha"])


class TestADFlow(unittest.TestCase):
    def setUp(self):

        prob = om.Problem()
        prob.model = Top()

        # DVs
        prob.model.add_design_var("alpha", lower=-5, upper=10, ref=10.0)
        prob.model.add_design_var("x_a", indices=[3, 14, 20, 9], lower=-5, upper=10, ref=10.0)

        prob.model.add_constraint("mp.s0.aero_funcs.cl", ref=1.0, equals=0.5)
        prob.model.add_constraint("mp.s0.aero_funcs.cd", ref=1.0, equals=0.5)

        prob.setup()

        self.prob = prob
        # om.n2(
        #     prob,
        #     show_browser=False,
        #     outfile="test_aero_derivs.html" ,
        # )

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()

        data = self.prob.check_totals(wrt="alpha", step=1e-8, step_calc="rel")
        for var, err in data.items():
            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 1e-3)

        data = self.prob.check_totals(wrt="x_a", step=1e-8)
        for var, err in data.items():
            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 5e-3)



if __name__ == "__main__":
    unittest.main()