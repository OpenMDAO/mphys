#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from omfsi.mphy_multipoint import MPHY_Multipoint

# these imports will be from the respective codes' repos rather than omfsi
from omfsi.adflow_component_configure import ADflow_builder
from omfsi.dvgeo_component_configure import OM_DVGEOCOMP

from baseclasses import *
from tacs import elements, constitutive, functions

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
            # 'ankswitchtol':1e-1,
            'nsubiterturb': 5,

            # NK Solver Parameters
            'useNKSolver':True,
            'nkswitchtol':1e-4,

            # Termination Criteria
            'L2Convergence':1e-14,
            'L2ConvergenceCoarse':1e-2,
            # 'L2ConvergenceRel': 1e-4,
            'nCycles':10000,

            # force integration
            'forcesAsTractions':False,
        }

        adflow_builder = ADflow_builder(aero_options)


        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # add the geometry component
        self.add_subsystem('geo', OM_DVGEOCOMP(ffd_file='ffd.xyz'))

        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'mp_group',
            MPHY_Multipoint(
                aero_builder   = adflow_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphy_add_scenario('s0')

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

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.mp_group.s0.aero.mphy_set_ap(ap0)

        # create geometric DV setup
        points = self.mp_group.mphy_add_coordinate_input()
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
        self.dvs.add_output('alpha', val=1.5)
        self.dvs.add_output('twist', val=np.array([0]*nTwist))

        self.connect('alpha', 'mp_group.s0.aero.alpha')
        self.connect('twist', 'geo.twist')

        # define the design variables
        self.add_design_var('alpha', lower=   0.0, upper=10.0, scaler=0.1)
        self.add_design_var('twist', lower= -10.0, upper=10.0, scaler=0.01)

        # add constraints and the objective
        self.add_constraint('mp_group.s0.aero.funcs.cl', equals=0.5, scaler=10.0)
        self.add_objective('mp_group.s0.aero.funcs.cd', scaler=100.0)

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile='mphy_aero.html')
prob.run_model()
# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('cl =',prob['mp_group.s0.aero.funcs.cl'])
    print('cd =',prob['mp_group.s0.aero.funcs.cd'])
