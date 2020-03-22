#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from omfsi.mphy_multipoint import MPHY_Multipoint

# these imports will be from the respective codes' repos rather than omfsi
from omfsi.adflow_component_configure import ADflow_builder
from omfsi.tacs_component_configure import TACS_builder
from omfsi.meld_xfer_component_configure import MELD_builder
from omfsi.rlt_xfer_component_configure import RLT_builder

from baseclasses import *
from tacs import elements, constitutive, functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

# flag to use meld (False for RLT)
use_meld = True
# use_meld = False

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
            evalFuncs=['lift','drag', 'cl', 'cd']
        )
        ap0.addDV('alpha',value=1.5,name='alpha')
        ap0.addDV('mach',value=0.8,name='mach')

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.mp_group.s0.aero.mphy_set_ap(ap0)

        # define the aero DVs in the IVC
        # s0
        self.dvs.add_output('alpha0', val=1.5)
        self.dvs.add_output('mach0', val=0.8)

        # connect to the aero for each scenario
        self.connect('alpha0', 'mp_group.s0.aero.alpha')
        self.connect('mach0', 'mp_group.s0.aero.mach')

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
