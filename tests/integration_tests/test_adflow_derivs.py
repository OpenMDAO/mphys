# --- Python 3.8 ---
"""
@File    :   test_aero_derivs.py
@Time    :   2020/12/17
@Author  :   Josh Anibal
@Desc    :   File for testing the derivatives of the mphys adflow wrapper
"""

import os
# === Standard Python modules ===
import unittest

# === Extension modules ===
import openmdao.api as om
from adflow import ADFLOW
from baseclasses import AeroProblem
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint
from mphys.scenarios.aerodynamic import ScenarioAerodynamic

baseDir = os.path.dirname(os.path.abspath(__file__))


class Top(Multipoint):
    def setup(self):

        ################################################################################
        # AERO
        ################################################################################

        from adflow.mphys import ADflowBuilder

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
            "writesurfacesolution": False,
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
            "adjointl2convergence": 1e-14,
        }

        adflow_builder = ADflowBuilder(aero_options, scenario="aerodynamic")
        adflow_builder.initialize(self.comm)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc for aero DVs
        self.add_subsystem("dv_aero", om.IndepVarComp(), promotes=["*"])
        # ivc for mesh coordinate dvs. we need a separate distributed component for the shape by conn stuff to work
        dv_coords = self.add_subsystem("dv_coords", om.IndepVarComp(), promotes=["*"])

        # TODO this only works for a serial run. must be updated for parallel
        x_a = adflow_builder.solver.getSurfaceCoordinates(groupName="allWalls").flatten()
        dv_coords.add_output("x_aero", val=x_a, distributed=False)

        # normally we would have a mesh comp but we just do the parallel ivc for the test.
        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=adflow_builder))
        self.connect("x_aero", "cruise.x_aero")

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
        ap0.addDV("alpha", value=2.0, name="aoa", units="deg")
        self.cruise.coupling.mphys_set_ap(ap0)
        self.cruise.aero_post.mphys_set_ap(ap0)

        self.dv_aero.add_output("aoa", val=alpha0, units="deg")
        self.connect("aoa", ["cruise.coupling.aoa", "cruise.aero_post.aoa"])


class TestADFlow(unittest.TestCase):

    # TODO keep this test to a single proc for now because we perturb mesh coordinates directly
    N_PROCS = 1

    def setUp(self):

        prob = om.Problem()
        prob.model = Top()

        # DVs
        prob.model.add_design_var("aoa", lower=-5, upper=10, ref=10.0, units="deg")
        prob.model.add_design_var("x_aero", indices=[3, 14, 20, 9], lower=-5, upper=10, ref=10.0)

        prob.model.add_constraint("cruise.aero_post.cl", ref=1.0, equals=0.5)
        prob.model.add_constraint("cruise.aero_post.cd", ref=1.0, equals=0.5)

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

        data = self.prob.check_totals(wrt="aoa", step=1e-8, step_calc="rel")
        for var, err in data.items():
            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 1e-3)

        data = self.prob.check_totals(wrt="x_aero", step=1e-8)
        for var, err in data.items():
            rel_err = err["rel error"]
            assert_near_equal(rel_err.forward, 0.0, 5e-3)


if __name__ == "__main__":
    unittest.main()
