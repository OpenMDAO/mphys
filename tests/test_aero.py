import numpy as np

import openmdao.api as om
from mphys.mphys_adflow import ADflow_builder
from mphys.mphys_adflow import ADflowGroup

from baseclasses import *
from mpi4py import MPI

class Top(om.Group):

    def setup(self):


        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        dvs.add_output('alpha', val=1.5)
        self.connect('alpha', 'aero.alpha')
        
        
        
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
            'L2Convergence':1e-2,
            'L2ConvergenceCoarse':1e-2,
            'nCycles':1000,
        }

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
        
        aero = ADflowGroup(aero_problem = ap0, 
                           solver_options = aero_options, 
                           group_options = {
                               'mesh': False,
                               'deformer': False
                           })

        # the solver must be created before added the group as a subsystem.

        aero.init_solver_object(self.comm)
        self.add_subsystem('aero', aero)

        # ivc to keep the top level DVs


        # self.aero.set_ap(ap0)



################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
# om.n2(prob, show_browser=False, outfile='mphys_aero.html')

prob.run_model()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)

# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('cl =',prob['aero.cl'])
    print('cd =',prob['aero.cd'])
