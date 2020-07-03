#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys.mphys_multipoint import MPHYS_Multipoint

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflow_builder
from mphys.mphys_dvgeo import OM_DVGEOCOMP

from baseclasses import *

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--task', default='run')
args = parser.parse_args()

class Top(om.Group):

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
            'writevolumesolution':False,
            'writesurfacesolution':False,

            # Physics Parameters
            'equationType':'RANS',

            # Solver Parameters
            'smoother':'dadi',
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

        adflow_builder = ADflow_builder(aero_options)


        ################################################################################
        # mphys setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # add the geometry component
        self.add_subsystem('geo', OM_DVGEOCOMP(ffd_file='ffd.xyz'))

        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'mp_group',
            MPHYS_Multipoint(
                aero_builder   = adflow_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')
        mp.mphys_add_scenario('s1')

        # add an exec comp to average two drags
        self.add_subsystem('drag', om.ExecComp('cd_out=(cd0+cd1)/2'))

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        ap0 = AeroProblem(
            name='ap0',
            mach=0.8,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['cl', 'cd']
        )
        ap0.addDV('alpha',value=1.5,name='alpha')

        ap1 = AeroProblem(
            name='ap1',
            mach=0.7,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['cl', 'cd']
        )
        ap1.addDV('alpha',value=1.5,name='alpha')

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.mp_group.s0.solver_group.aero.mphys_set_ap(ap0)
        self.mp_group.s0.aero_funcs.mphys_set_ap(ap0)
        self.mp_group.s1.solver_group.aero.mphys_set_ap(ap1)
        self.mp_group.s1.aero_funcs.mphys_set_ap(ap1)

        # create geometric DV setup
        points = self.mp_group.mphys_add_coordinate_input()
        # add these points to the geometry object
        self.geo.nom_add_point_dict(points)
        # connect
        for key in points:
            self.connect('geo.%s'%key, 'mp_group.%s'%key)

        # geometry setup

        # Create reference axis
        nRefAxPts = self.geo.nom_addRefAxis(name='wing', xFraction=0.25, alignIndex='k')
        nTwist = nRefAxPts - 1

        # Set up global design variables
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z['wing'].coef[i] = val[i-1]
        self.geo.nom_addGeoDVGlobal(dvName='twist', value=np.array([0]*nTwist), func=twist)

        # add dvs to ivc and connect
        self.dvs.add_output('alpha0', val=1.5)
        self.dvs.add_output('alpha1', val=1.5)
        self.dvs.add_output('twist', val=np.array([0]*nTwist))

        self.connect('alpha0', ['mp_group.s0.solver_group.aero.alpha', 'mp_group.s0.aero_funcs.alpha'])
        self.connect('alpha1', ['mp_group.s1.solver_group.aero.alpha', 'mp_group.s1.aero_funcs.alpha'])
        self.connect('twist', 'geo.twist')

        # define the design variables
        self.add_design_var('alpha0', lower=   0.0, upper=10.0, scaler=0.1)
        self.add_design_var('alpha1', lower=   0.0, upper=10.0, scaler=0.1)
        self.add_design_var('twist', lower= -10.0, upper=10.0, scaler=0.01)

        # add constraints and the objective
        self.add_constraint('mp_group.s0.aero_funcs.cl', equals=0.5, scaler=10.0)
        self.add_constraint('mp_group.s1.aero_funcs.cl', equals=0.5, scaler=10.0)

        # connect the two drags to drag average
        self.connect('mp_group.s0.aero_funcs.cd', 'drag.cd0')
        self.connect('mp_group.s1.aero_funcs.cd', 'drag.cd1')
        self.add_objective('drag.cd_out', scaler=100.0)

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.opt_settings ={
    'Major feasibility tolerance': 1e-4, #1e-4,
    'Major optimality tolerance': 1e-4, #1e-8,
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
om.n2(prob, show_browser=False, outfile='mphy_aero_2pt.html')

if args.task == 'run':
    prob.run_model()
elif args.task == 'opt':
    prob.run_driver()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)

# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('cl =',prob['mp_group.s0.aero_funcs.cl'])
    print('cd =',prob['mp_group.s0.aero_funcs.cd'])

    print("Scenario 1")
    print('cl =',prob['mp_group.s1.aero_funcs.cl'])
    print('cd =',prob['mp_group.s1.aero_funcs.cd'])
