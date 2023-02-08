import argparse

import openmdao.api as om
from mphys.multipoint import MultipointParallel
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from adflow.mphys import ADflowBuilder
from baseclasses import AeroProblem


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="run")
parser.add_argument("--level", type=str, default="L1")
args = parser.parse_args()


class ParallelCruises(MultipointParallel):
    def setup(self):

        ################################################################################
        # ADflow options
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
            "L2Convergence": 1e-12,
            "adjointl2convergence": 1e-12,
            "L2ConvergenceCoarse": 1e-2,
            # 'L2ConvergenceRel': 1e-4,
            "nCycles": 10000,
            "infchangecorrection": True,
            # force integration
            "forcesAsTractions": False,
        }

        adflow_builder = ADflowBuilder(aero_options, scenario="aerodynamic")
        adflow_builder.initialize(self.comm)

        self.mphys_add_scenario(
            "cruise0",
            ScenarioAerodynamic(aero_builder=adflow_builder, in_MultipointParallel=True),
        )

        self.mphys_add_scenario(
            "cruise1",
            ScenarioAerodynamic(aero_builder=adflow_builder, in_MultipointParallel=True),
        )


class Top(om.Group):
    def setup(self):

        ################################################################################
        # mphys setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the parallel multipoint group
        self.add_subsystem("mp", ParallelCruises())

        # add an exec comp to average two drags
        self.add_subsystem("drag", om.ExecComp("cd_out=(cd0+cd1)/2"))

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        aoa = 1.5
        ap0 = AeroProblem(
            name="ap0", mach=0.8, altitude=10000, alpha=aoa, areaRef=45.5, chordRef=3.25, evalFuncs=["cl", "cd"]
        )
        ap0.addDV("alpha", value=aoa, name="aoa", units="deg")

        ap1 = AeroProblem(
            name="ap1", mach=0.7, altitude=10000, alpha=1.5, areaRef=45.5, chordRef=3.25, evalFuncs=["cl", "cd"]
        )
        ap1.addDV("alpha", value=aoa, name="aoa", units="deg")

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        try:
            self.mp.cruise0.coupling.mphys_set_ap(ap0)
            self.mp.cruise0.aero_post.mphys_set_ap(ap0)
        except AttributeError:
            pass

        try:
            self.mp.cruise1.coupling.mphys_set_ap(ap1)
            self.mp.cruise1.aero_post.mphys_set_ap(ap1)
        except AttributeError:
            pass

        # create geometric DV setup
        # points = self.mesh.mphys_get_surface_mesh()

        # add pointset
        # self.geometry.nom_add_discipline_coords("aero", points)

        # add these points to the geometry object
        # self.geo.nom_add_point_dict(points)
        # create constraint DV setup
        # tri_points = self.mesh.mphys_get_triangulated_surface()
        # self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup

        # Create reference axis
        # nRefAxPts = self.geometry.nom_addRefAxis(name="wing", xFraction=0.25, alignIndex="k")
        # nTwist = nRefAxPts - 1

        # Set up global design variables
        # def twist(val, geo):
        #     for i in range(1, nRefAxPts):
        #         geo.rot_y["wing"].coef[i] = val[i - 1]

        # self.geometry.nom_addGeoDVGlobal(dvName="twist", value=np.array([0] * nTwist), func=twist)

        # add dvs to ivc and connect
        self.dvs.add_output("aoa0", val=aoa, units="deg")
        self.dvs.add_output("aoa1", val=aoa, units="deg")
        # self.dvs.add_output("twist", val=np.array([0] * nTwist))

        # TODO this is working but not the correct way to do it. the sensitivities are also wrong now.
        self.connect("aoa0", ["mp.cruise0.coupling.aoa", "mp.cruise0.aero_post.aoa"], src_indices=[0])
        self.connect("aoa1", ["mp.cruise1.coupling.aoa", "mp.cruise1.aero_post.aoa"], src_indices=[0])
        # self.connect("twist", "geometry.twist")

        # define the design variables
        self.add_design_var("aoa0", lower=0.0, upper=10.0, scaler=0.1, units="deg")
        self.add_design_var("aoa1", lower=0.0, upper=10.0, scaler=0.1, units="deg")
        # self.add_design_var("twist", lower=-10.0, upper=10.0, scaler=0.01)

        # add constraints and the objective
        self.add_constraint("mp.cruise0.aero_post.cl", equals=0.5, scaler=10.0)
        self.add_constraint("mp.cruise1.aero_post.cl", equals=0.5, scaler=10.0)

        # connect the two drags to drag average
        self.connect("mp.cruise0.aero_post.cd", "drag.cd0")
        self.connect("mp.cruise1.aero_post.cd", "drag.cd1")
        self.add_objective("drag.cd_out", scaler=100.0)


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SNOPT"
prob.driver.opt_settings = {
    "Major feasibility tolerance": 1e-4,  # 1e-4,
    "Major optimality tolerance": 1e-4,  # 1e-8,
    "Verify level": 0,
    "Major iterations limit": 200,
    "Minor iterations limit": 1000000,
    "Iterations limit": 1500000,
    "Nonderivative linesearch": None,
    "Major step limit": 0.01,
    "Function precision": 1.0e-8,
    # 'Difference interval':1.0e-6,
    # 'Hessian full memory':None,
    "Hessian frequency": 200,
    # 'Linesearch tolerance':0.99,
    "Print file": "SNOPT_print.out",
    "Summary file": "SNOPT_summary.out",
    "Problem Type": "Minimize",
    # 'New superbasics limit':500,
    "Penalty parameter": 1.0,
}

# prob.driver.options['debug_print'] = ['totals', 'desvars']

prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys_aero_2pt_parallel.html")

if args.task == "run":
    prob.run_model()
elif args.task == "opt":
    prob.run_driver()

# TODO list i/o did not work for parallel multipoint
# prob.model.list_inputs(units=True)
# prob.model.list_outputs(units=True)

# prob.model.list_outputs()

cl0 = prob.get_val("mp.cruise0.aero_post.cl", get_remote=True)
cd0 = prob.get_val("mp.cruise0.aero_post.cd", get_remote=True)

cl1 = prob.get_val("mp.cruise1.aero_post.cl", get_remote=True)
cd1 = prob.get_val("mp.cruise1.aero_post.cd", get_remote=True)

if prob.model.comm.rank == 0:
    print("Cruise 0")
    print("cl =", cl0)
    print("cd =", cd1)

    print("Cruise 1")
    print("cl =", cl1)
    print("cd =", cd1)
