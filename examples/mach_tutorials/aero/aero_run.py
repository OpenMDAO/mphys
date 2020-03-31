import numpy as np

import openmdao.api as om
from mphys.mphys_multipoint import MPHYS_Multipoint
from mphys.mphys_adflow import ADflow_builder
from baseclasses import *
from mpi4py import MPI

class Top(om.Group):

    def setup(self):

        aero_options = {
            # I/O Parameters
            'gridFile':'wing_vol.cgns',
            'outputDirectory':'.',
            'monitorvariables':['resrho','cl','cd'],
            'writeTecplotSurfaceSolution':True,

            # Physics Parameters
            'equationType':'RANS',

            # Solver Parameters
            'smoother':'dadi',
            'CFL':0.5,
            'CFLCoarse':0.25,
            'MGCycle':'sg',
            'MGStartLevel':-1,
            'nCyclesCoarse':250,

            # ANK Solver Parameters
            'useANKSolver':True,
            'nsubiterturb': 5,

            # NK Solver Parameters
            'useNKSolver':True,
            'nkswitchtol':1e-4,

            # Termination Criteria
            'L2Convergence':1e-12,
            'L2ConvergenceCoarse':1e-2,
            'nCycles':1000,
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
            MPHYS_Multipoint(
                aero_builder   = adflow_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

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
            evalFuncs=['cl','cd']
        )
        ap0.addDV('alpha',value=1.5,name='alpha')

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.mp_group.s0.aero.mphys_set_ap(ap0)

        # add dvs to ivc and connect
        self.dvs.add_output('alpha', val=1.5)

        self.connect('alpha', 'mp_group.s0.aero.alpha')

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
# om.n2(prob, show_browser=False, outfile='mphy_aero.html')

prob.run_model()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)

# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('cl =',prob['mp_group.s0.aero.funcs.cl'])
    print('cd =',prob['mp_group.s0.aero.funcs.cd'])
