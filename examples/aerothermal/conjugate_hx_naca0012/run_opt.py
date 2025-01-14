# from mpi4py import MPI
import argparse

import numpy as np
import openmdao.api as om
from adflow.mphys import ADflowBuilder
from baseclasses import AeroProblem
from funtofem.mphys import MeldThermalBuilder
from pygeo.mphys import OM_DVGEOCOMP
from tacs import constitutive, elements, functions
from tacs.mphys import TacsBuilder

from mphys import MPhysVariables, Multipoint
from mphys.scenarios import ScenarioAeroThermal

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="run")
parser.add_argument("--complexify", action="store_true")
args = parser.parse_args()

complexify = args.complexify


def add_elements(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):

    # Create the constitutvie propertes and model
    props = constitutive.MaterialProperties(kappa=230)
    # con = constitutive.PlaneStressConstitutive(props)
    con = constitutive.SolidConstitutive(props)
    heat = elements.HeatConduction3D(con)

    # Create the basis class
    # quad_basis = elements.LinearQuadBasis()
    basis = elements.LinearHexaBasis()

    # Create the element
    elem = elements.Element3D(heat, basis)

    # Loop over components, creating stiffness and element object for each
    # num_components = mesh.getNumComponents()
    # for i in range(num_components):
    #     descriptor = mesh.getElementDescript(i)
    #     print("Setting element with description %s" % (descriptor))
    #     mesh.setElement(i, element)

    # ndof = heat.getVarsPerNode()

    return elem


def get_surface_mapping(Xpts_array):
    Xpts_array = Xpts_array.reshape(len(Xpts_array) // 3, 3)
    unique_x = set(Xpts_array[:, 0])
    unique_x = list(unique_x)
    unique_x.sort()

    plate_surface = []
    mask = []
    lower_mask = []
    upper_mask = []

    for x in unique_x:
        mask_sec = np.where(Xpts_array[:, 0] == x)[0]

        # find min and max y points
        max_mask = np.where(Xpts_array[mask_sec, 1] == np.max(Xpts_array[mask_sec, 1]))[0]
        min_mask = np.where(Xpts_array[mask_sec, 1] == np.min(Xpts_array[mask_sec, 1]))[0]

        lower_mask.extend(mask_sec[min_mask])
        upper_mask.extend(mask_sec[max_mask])

        # mask.extend(mask_sec[min_mask], mask_sec[max_mask])

        # plate_surface.extend([lower_mask, upper_mask])
        # mapping.extend

    lower_mask = np.array(lower_mask)
    upper_mask = np.array(upper_mask)
    mask = np.concatenate((lower_mask, upper_mask))
    mapping = mask
    # plate_surface = np.array(Xpts_array[mask])

    return mapping


class Top(Multipoint):
    def setup(self):
        # ivc to keep the top level DVs
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])


        ################################################################################
        # ADflow Setup
        ################################################################################
        aero_options = {
            # I/O Parameters
            "gridFile": "./meshes/n0012_hot_000_vol.cgns",
            # "restartfile": "./meshes/n0012_hot_000_vol.cgns",
            "outputDirectory": "./output",
            "monitorvariables": ["resrho", "cd", "cl", "totheattransfer"],
            "surfacevariables": ["cp", "vx", "vy", "vz", "mach", "heatflux", "temp"],
            # 'isovariables': ['shock'],
            "writeTecplotSurfaceSolution": False,
            "writevolumesolution": False,
            # 'writesurfacesolution':False,
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
            "useNKSolver": True,
            "nkswitchtol": 1e-3,
            # Termination Criteria
            "L2Convergence": 1e-13,
            "L2ConvergenceCoarse": 1e-2,
            "nCycles": 400,
            "adjointl2convergence": 1e-13,
            "heatxferratesAsFluxes": True,
            "printIterations": True,
            "liftIndex": 2,
            # "rkreset": True,
            # 'nRKReset': 50,
            # "rkreset": True,
        }

        aero_builder = ADflowBuilder(aero_options, scenario="aerothermal", complexify=complexify)
        aero_builder.initialize(self.comm)
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem(surface_groups=["allIsothermalWalls"]))

        # self.connect("mesh_aero.x_aero0", "cruise.x_aero")
        self.connect("mesh_aero.x_aero_allIsothermalWalls0", "cruise.x_aero_surface0")

        ################################################################################
        # TACS Setup
        ################################################################################
        thermal_builder = TacsBuilder(mesh_file="./meshes/n0012_hexa.bdf", element_callback=add_elements,
                                      discipline="thermal", surface_mapper=get_surface_mapping)
        thermal_builder.initialize(self.comm)
        ndv_struct = thermal_builder.get_ndv()

        self.add_subsystem("mesh_thermal", thermal_builder.get_mesh_coordinate_subsystem())
        self.connect("mesh_thermal.x_thermal_surface0", "cruise.x_thermal_surface0")
        ################################################################################
        # Transfer Scheme Setup
        ################################################################################

        thermalxfer_builder = MeldThermalBuilder(aero_builder, thermal_builder, isym=1, n=10, beta=0.5)
        thermalxfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS Setup
        ################################################################################

        self.add_subsystem("geometry", OM_DVGEOCOMP(file="./ffds/n0012_ffd.xyz", type="ffd", options= {"complex":complexify}))

        scenario = "cruise"

        self.mphys_add_scenario(
            scenario,
            ScenarioAeroThermal(
                aero_builder=aero_builder, thermal_builder=thermal_builder, thermalxfer_builder=thermalxfer_builder
            ),
            om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-12, atol=1e-14),
            om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-12, atol=1e-14),
        )

        self.connect(f"mesh_aero.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}",
                     f"geometry.{MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT}")
        self.connect(f"geometry.{MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT}",
                     f"cruise.{MPhysVariables.Aerodynamics.Surface.COORDINATES}")

        self.connect(f"mesh_thermal.{MPhysVariables.Thermal.Mesh.COORDINATES}",
                     f"geometry.{MPhysVariables.Thermal.Geometry.COORDINATES_INPUT}")
        self.connect(f"geometry.{MPhysVariables.Thermal.Geometry.COORDINATES_OUTPUT}",
                     f"cruise.{MPhysVariables.Thermal.COORDINATES}")

    def configure(self):
        # create the aero problems for the analysis point.
        # this is custom to the ADflow based approach we chose here.
        super().configure()

        aoa0 = 0.0
        ap0 = AeroProblem(
            name="naca0012_hot",
            V=60,  # m/s
            T=273 + 60,  # kelvin
            P=93e3,  # pa
            areaRef=1.0,  # m^2
            chordRef=1.0,  # m^2
            evalFuncs=["cd", "cl", "totheattransfer", "havg"],
            alpha=5.0,
            beta=0.00,
            xRef=0.0,
            yRef=0.0,
            zRef=0.0,
        )
        ap0.addDV("alpha", value=aoa0, name="aoa", units="deg")


        self.cruise.coupling.aero.mphys_set_ap(ap0)
        self.cruise.aero_post.mphys_set_ap(ap0)

        # define the aero DVs in the IVC
        self.dvs.add_output("aoa", val=aoa0, units="deg")

        # connect to the aero for each scenario
        self.connect("aoa", ["cruise.coupling.aero.aoa", "cruise.aero_post.aoa"])


        self.cruise.aero_post.mphys_add_BCDVs("Temperature", famGroup="allIsothermalWalls", dv_name="T_convect", coupling=True)
        self.cruise.coupling.aero.mphys_add_BCDVs("Temperature", famGroup="allIsothermalWalls", dv_name="T_convect", coupling=True)

        # new coupling variables were added so we need to reinitialize promotion
        self.cruise.coupling._mphys_promote_coupling_variables()

        # This should be done automatically in the future, but alas it is not yet.
        self.connect("cruise.T_convect", ["cruise.aero_post.T_convect"])

        # create geometric DV setup
        # points_aero = self.mesh_aero.mphys_get_surface_mesh()
        # points_thermal = self.mesh_thermal.mphys_get_mesh()

        # add pointset
        self.geometry.nom_add_discipline_coords("aero")
        self.geometry.nom_add_discipline_coords("thermal")

        # add these points to the geometry object
        # self.geo.nom_add_point_dict(points)
        # create constraint DV setup
        tri_points = self.mesh_aero.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # geometry setup

        # Create reference axis

        nLocal = self.geometry.nom_addSpanwiseLocalDV("shape", 'k', axis="y")

        le = 0.01
        leList = [[le, 0, le], [le, 0, 1.0 - le]]
        teList = [[1.0 - le, 0, le], [1.0 - le, 0, 1.0 - le]]

        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList, nSpan=20, nChord=20)
        self.geometry.nom_add_LETEConstraint(
            "lecon",
            0,
            "iLow",
            topID='j'
        )
        self.geometry.nom_add_LETEConstraint("tecon", 0, "iHigh", topID='j')
        # add dvs to ivc and connect
        self.dvs.add_output("shape", val=np.array([0] * nLocal))
        self.connect("shape", ["geometry.shape"])




################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()


model = prob.model
# openmdao.utils.general_utils.ignore_errors(True)
# prob.model._set_subsys_connection_errors(False)
# prob.model._raise_connection_errors = False

prob.driver = om.pyOptSparseDriver()
prob.driver.hist_file = "./opt_hist.db"
prob.driver.options["optimizer"] = "SNOPT"
prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]
prob.driver.opt_settings = {
    "Major feasibility tolerance": 1e-4,  # 1e-4,
    "Major optimality tolerance": 1e-8,  # 1e-8,
    "Verify level": 3,
    "Major iterations limit": 600,
    "Minor iterations limit": 1000000,
    "Iterations limit": 1500000,
    # 'Nonderivative linesearch':None,
    "Major step limit": 1e-0,
    'Function precision':1.0e-8,
    # 'Difference interval':1.0e-6,
    "Hessian full memory": None,
    "Hessian frequency": 200,
    "Hessian updates": 200,
    # 'Linesearch tolerance':0.99,
    "Print file": "./SNOPT_print.out",
    "Summary file": "./summary_SNOPT.out",
    # 'New superbasics limit':500,
    "Penalty parameter": 1.0,  # 1.0,
}


prob.model.add_design_var("shape", lower=-0.05, upper=0.05, )
# prob.model.add_design_var("x_thermal0", lower=-0.05, upper=0.05, indices=[10])
# prob.model.add_design_var("x_thermal0", lower=-0.05, upper=0.05, indices=[1])
# prob.model.add_design_var("aoa", lower=-5.0, upper=10.0)

prob.model.add_constraint("geometry.lecon", equals=0)
prob.model.add_constraint("geometry.tecon", equals=0)
prob.model.add_constraint("geometry.thickcon", lower=0.3)
prob.model.add_constraint("geometry.volcon", lower=0.3)

# prob.model.add_constraint("geometry.x_thermal0",  equals=0.0)
# prob.model.add_constraint("cruise.q_conduct",  indices=[1])
# prob.model.add_constraint("cruise.q_convect",  indices=[1])

# prob.model.add_objective("cruise.T_convect",  index=1)
# prob.model.add_objective("cruise.T_convect",  index=1)
# prob.model.add_objective("cruise.q_conduct_surf2vol.q_conduct",  index=1)
# prob.model.add_objective("cruise.q_conduct",  index=1)
prob.model.add_objective("cruise.aero_post.totheattransfer",  ref=3e3)
prob.model.add_constraint("cruise.aero_post.cd", upper=1.0*2e-2, ref=2e-2)
prob.model.add_constraint("cruise.aero_post.cl", equals=0.5)

# prob.driver.recording_options["record_inputs"] = False
# prob.driver.recording_options["record_desvars"] = True
# prob.driver.recording_options["record_responses"] = False
# prob.driver.recording_options["record_objectives"] = True



# prob.setup(mode="rev")

# om.n2(prob, show_browser=False, outfile="mphys_at_adflow_tacs_meld.html")
if args.task == "run":
    prob.setup(mode="rev", force_alloc_complex=True)
    # om.n2(prob, show_browser=False, outfile="mphys_at_adflow_tacs_meld_tmp.html")
    prob.run_model()

    if complexify:
        prob.model.approx_totals(method='cs', step=1e-40)
        from openmdao.core.total_jac import _TotalJacInfo

        total_info = _TotalJacInfo(prob, "cruise.aero_post.totheattransfer", "shape", False, return_format='flat_dict', approx=True,
                                    driver_scaling=False)
        Jfd = total_info.compute_totals_approx(initialize=True)
        print('------------------------------------')
        print('J cs = ', Jfd)
        print('------------------------------------')
    else:
        # from openmdao.core.total_jac import _TotalJacInfo
        # prob._mode='rev'
        # total_info = _TotalJacInfo(prob, "cruise.T_conduct", "x_thermal0", False, return_format='flat_dict', approx=False,
        #                     driver_scaling=False)
        # Jfd = total_info.compute_totals()
        # print('------------------------------------')
        # print('J adjoint = ', Jfd)
        # print('------------------------------------')
        prob.check_totals(step=1e-10)



    # prob.model.list_outputs(print_arrays=True)
    # prob.check_partials(compact_print=True, includes='*geometry*')
elif args.task == "opt":
    prob.setup(mode="rev")


    prob.run_driver()
