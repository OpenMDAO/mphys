import numpy as np
import argparse

import openmdao.api as om
from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural

from adflow.mphys import ADflowBuilder
from baseclasses import AeroProblem
from tacs.mphys import TacsBuilder
from mphys.solver_builders.mphys_meld import MeldBuilder
# TODO RLT needs to be updated with the new tacs wrapper
# from rlt.mphys import RltBuilder

import tacs_setup


parser = argparse.ArgumentParser()
parser.add_argument("--xfer", default="meld", choices=["meld", "rlt"])
parser.add_argument("--level", type=str, default="L1")
args = parser.parse_args()

if args.xfer == "meld":
    forcesAsTractions = False
else:
    forcesAsTractions = True


class Top(Multipoint):
    def setup(self):

        ################################################################################
        # ADflow Setup
        ################################################################################
        aero_options = {
            # I/O Parameters
            "gridFile": f"wing_vol_{args.level}.cgns",
            "outputDirectory": ".",
            "monitorvariables": ["resrho", "resturb", "cl", "cd"],
            "writeTecplotSurfaceSolution": False,
            # 'writevolumesolution':False,
            # 'writesurfacesolution':False,
            # Physics Parameters
            "equationType": "RANS",
            "liftindex": 3,  # z is the lift direction
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
            "forcesAsTractions": forcesAsTractions,
        }

        aero_builder = ADflowBuilder(aero_options, scenario="aerostructural")
        aero_builder.initialize(self.comm)
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # TACS Setup
        ################################################################################
        tacs_options = {
            "element_callback": tacs_setup.element_callback,
            "problem_setup": tacs_setup.problem_setup,
            "mesh_file": "wingbox.bdf",
        }

        struct_builder = TacsBuilder(tacs_options)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Transfer Scheme Setup
        ################################################################################

        if args.xfer == "meld":
            isym = 1  # y-symmetry
            ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)
            ldxfer_builder.initialize(self.comm)
        else:
            # or we can use RLT:
            xfer_options = {"transfergaussorder": 2}
            ldxfer_builder = RltBuilder(xfer_options, aero_builder, struct_builder)
            ldxfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS Setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        dvs.add_output("dv_struct", np.array(ndv_struct * [0.002]))

        for iscen, scenario in enumerate(["cruise", "maneuver"]):
            nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
            linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
            self.mphys_add_scenario(
                scenario,
                ScenarioAeroStructural(
                    aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=ldxfer_builder
                ),
                nonlinear_solver,
                linear_solver,
            )

            for discipline in ["aero", "struct"]:
                self.mphys_connect_scenario_coordinate_source("mesh_%s" % discipline, scenario, discipline)

            self.connect("dv_struct", f"{scenario}.dv_struct")

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        aoa0 = 2.0
        ap0 = AeroProblem(
            name="cruise",
            mach=0.85,
            altitude=10000,
            alpha=aoa0,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=["lift", "drag", "cl", "cd"],
        )
        ap0.addDV("alpha", value=aoa0, name="aoa", units="deg")

        aoa1 = 5.0
        ap1 = AeroProblem(
            name="maneuver",
            mach=0.85,
            altitude=10000,
            alpha=aoa1,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=["lift", "drag", "cl", "cd"],
        )
        ap1.addDV("alpha", value=aoa1, name="aoa", units="deg")

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.cruise.coupling.aero.mphys_set_ap(ap0)
        self.cruise.aero_post.mphys_set_ap(ap0)

        self.maneuver.coupling.aero.mphys_set_ap(ap1)
        self.maneuver.aero_post.mphys_set_ap(ap1)

        # define the aero DVs in the IVC
        self.dvs.add_output("aoa0", val=aoa0, units="deg")
        self.dvs.add_output("aoa1", val=aoa1, units="deg")

        # connect to the aero for each scenario
        self.connect("aoa0", ["cruise.coupling.aero.aoa", "cruise.aero_post.aoa"])
        self.connect("aoa1", ["maneuver.coupling.aero.aoa", "maneuver.aero_post.aoa"])


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile="mphys_as_adflow_tacs_%s_2pt.html" % args.xfer)
prob.run_model()

prob.model.list_outputs()

if prob.model.comm.rank == 0:
    print("Cruise")
    print("cl =", prob["cruise.aero_post.cl"])
    print("cd =", prob["cruise.aero_post.cd"])

    print("Maneuver")
    print("cl =", prob["maneuver.aero_post.cl"])
    print("cd =", prob["maneuver.aero_post.cd"])
