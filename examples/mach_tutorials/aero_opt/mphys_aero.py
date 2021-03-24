import numpy as np
from mpi4py import MPI

import openmdao.api as om
from mphys.multipoint import Multipoint
from mphys.mphys_adflow import ADflowBuilder
from baseclasses import AeroProblem
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from mphys.mphys_dvgeo import OM_DVGEOCOMP

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--task', default='run')
args = parser.parse_args()

class Top(Multipoint):

    def setup(self):

        ################################################################################
        # ADflow options
        ################################################################################
        aero_options = {
            # I/O Parameters
            'gridFile':'wing_vol.cgns',
            'outputDirectory':'.',
            'monitorvariables':['resrho','resturb','cl','cd'],
            'writeTecplotSurfaceSolution':False,
            # 'writevolumesolution':False,
            # 'writesurfacesolution':False,

            # Physics Parameters
            'equationType':'RANS',

            # Solver Parameters
            'smoother':'DADI',
            'CFL':1.5,
            'CFLCoarse':1.25,
            'MGCycle':'sg',
            'MGStartLevel':-1,
            'nCyclesCoarse':250,

            # ANK Solver Parameters
            'useANKSolver':True,
            'nsubiterturb': 5,
            'anksecondordswitchtol':1e-4,
            'ankcoupledswitchtol': 1e-6,
            'ankinnerpreconits':2,
            'ankouterpreconits':2,
            'anklinresmax': 0.1,

            # Termination Criteria
            'L2Convergence':1e-12,
            'adjointl2convergence':1e-12,
            'L2ConvergenceCoarse':1e-2,
            # 'L2ConvergenceRel': 1e-4,
            'nCycles':10000,
            'infchangecorrection':True,

            # force integration
            'forcesAsTractions':False,
        }

        adflow_builder = ADflowBuilder(aero_options, scenario="aerodynamic")
        adflow_builder.initialize(self.comm)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc to keep the top level DVs
        self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # create the mesh component
        self.add_subsystem("mesh", adflow_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem('geometry', OM_DVGEOCOMP(ffd_file='ffd.xyz'))

        self.mphys_add_scenario("cruise", ScenarioAerodynamic(aero_builder=adflow_builder))

        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "cruise.x_aero")

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        aoa = 1.5
        ap0 = AeroProblem(
            name='ap0',
            mach=0.8,
            altitude=10000,
            alpha=aoa,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['cl', 'cd']
        )
        ap0.addDV('alpha', value=aoa, name='aoa', units='deg')

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.cruise.coupling.mphys_set_ap(ap0)
        self.cruise.aero_post.mphys_set_ap(ap0)

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
        nRefAxPts = self.geometry.nom_addRefAxis(name='wing', xFraction=0.25, alignIndex='k')
        nTwist = nRefAxPts - 1

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z['wing'].coef[i] = val[i-1]
        self.geometry.nom_addGeoDVGlobal(dvName='twist', value=np.array([0]*nTwist), func=twist)
        nLocal = self.geometry.nom_addGeoDVLocal(dvName='thickness')

        leList = [[0.01, 0, 0.001], [7.51, 0, 13.99]]
        teList = [[4.99, 0, 0.001], [8.99, 0, 13.99]]
        self.geometry.nom_addThicknessConstraints2D('thickcon', leList, teList, nSpan=10, nChord=10)
        self.geometry.nom_addVolumeConstraint('volcon', leList, teList, nSpan=20, nChord=20)
        nLECon = self.geometry.nom_add_LETEConstraint('lecon', 0, 'iLow',)
        nTECon = self.geometry.nom_add_LETEConstraint('tecon', 0, 'iHigh')
        # add dvs to ivc and connect
        self.dvs.add_output('aoa', val=aoa, units='deg')
        self.dvs.add_output('local', val=np.array([0]*nLocal))
        self.dvs.add_output('twist', val=np.array([0]*nTwist))

        self.connect('aoa', ["cruise.coupling.aoa", "cruise.aero_post.aoa"])
        self.connect('local', 'geometry.thickness')
        self.connect('twist', 'geometry.twist')

        # define the design variables
        self.add_design_var('aoa', lower=   0.0, upper=10.0, scaler=0.1, units='deg')
        self.add_design_var('local', lower= -0.5, upper=0.5, scaler=0.01)
        self.add_design_var('twist', lower= -10.0, upper=10.0, scaler=0.01)

        # add constraints and the objective
        self.add_constraint('cruise.aero_post.cl', equals=0.5, scaler=10.0)
        self.add_constraint('geometry.thickcon', lower=1.0, scaler=1.0)
        self.add_constraint('geometry.volcon', lower=1.0, scaler=1.0)
        self.add_constraint('geometry.tecon', equals=0.0, scaler=1.0, linear=True)
        self.add_constraint('geometry.lecon', equals=0.0, scaler=1.0, linear=True)
        self.add_objective('cruise.aero_post.cd', scaler=100.0)

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.opt_settings ={
    'Major feasibility tolerance': 1e-4, #1e-4,
    'Major optimality tolerance': 1e-3, #1e-8,
    'Verify level': 0,
    'Major iterations limit':200,
    'Minor iterations limit':1000000,
    'Iterations limit':1500000,
    'Nonderivative linesearch':None,
    'Major step limit': 0.01,
    'Function precision':1.0e-8,
    # 'Difference interval':1.0e-6,
    # 'Hessian full memory':None,
    'Hessian frequency' : 200,
    #'Linesearch tolerance':0.99,
    'Print file':'SNOPT_print.out',
    'Summary file': 'SNOPT_summary.out',
    'Problem Type':'Minimize',
    #'New superbasics limit':500,
    'Penalty parameter':1.0}

# prob.driver.options['debug_print'] = ['totals', 'desvars']

prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='mphys_aero.html')

if args.task == 'run':
    prob.run_model()
    # prob.model.list_outputs(print_arrays=True)
    # prob.check_partials(compact_print=True, includes='*geometry*')
    # prob.check_totals(compact_print=True)
elif args.task == 'opt':
    prob.run_driver()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)

# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('cl =',prob['cruise.aero_post.cl'])
    print('cd =',prob['cruise.aero_post.cd'])
