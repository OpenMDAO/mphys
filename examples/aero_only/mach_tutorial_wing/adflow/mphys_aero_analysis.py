import argparse

import openmdao.api as om
from adflow.mphys import ADflowBuilder
from baseclasses import AeroProblem

from mphys import MPhysVariables, Multipoint
from mphys.scenarios import ScenarioAerodynamic

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=str, default="L1")
args = parser.parse_args()


class Top(Multipoint):
    def setup(self):

        aero_options = {
            # I/O Parameters
            "gridFile": f"wing_vol_{args.level}.cgns",
            "outputDirectory": ".",
            "monitorvariables": ["resrho", "cl", "cd"],
            "writeTecplotSurfaceSolution": True,
            # Physics Parameters
            "equationType": "RANS",
            "liftindex": 3,  # z is the lift direction
            # Solver Parameters
            "smoother": "DADI",
            "CFL": 0.5,
            "CFLCoarse": 0.25,
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
            "L2Convergence": 1e-12,
            "L2ConvergenceCoarse": 1e-2,
            "nCycles": 1000,
        }

        adflow_builder = ADflowBuilder(aero_options, scenario="aerodynamic")
        adflow_builder.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the mesh and cruise scenario because we only have one analysis point
        self.add_subsystem("mesh", adflow_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario(
            "cruise", ScenarioAerodynamic(aero_builder=adflow_builder)
        )
        self.connect(
            f"mesh.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}",
            "cruise.x_aero",
        )

    def configure(self):
        aoa = 1.5
        ap0 = AeroProblem(
            name="ap0",
            mach=0.8,
            altitude=10000,
            alpha=aoa,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=["cl", "cd"],
        )
        ap0.addDV("alpha", value=aoa, name="aoa", units="deg")

        # set the aero problem in the coupling and post coupling groups
        self.cruise.coupling.mphys_set_ap(ap0)
        self.cruise.aero_post.mphys_set_ap(ap0)

        # add dvs to ivc and connect
        self.dvs.add_output("aoa", val=aoa, units="deg")

        # call the promote inputs to propagate aoa dvs
        # TODO does not work now
        # self.cruise._mphys_promote_inputs()
        # so connect manually
        self.connect("aoa", ["cruise.coupling.aoa", "cruise.aero_post.aoa"])


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
om.n2(prob, show_browser=False, outfile="mphys_aero.html")

prob.run_model()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)

# prob.model.list_outputs()
if prob.model.comm.rank == 0:
    print("Scenario 0")
    print("cl =", prob["cruise.aero_post.cl"])
    print("cd =", prob["cruise.aero_post.cd"])
