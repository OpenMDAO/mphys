import numpy as np

import openmdao.api as om
from mphys.mphys_multipoint import ParallelMultipoint, SerialMultipoint
from mphys.mphys_adflow import ADflow_builder
from mphys.mphys_adflow import ADflowGroup

from baseclasses import *
from mpi4py import MPI

class Top(om.Group):

    def setup(self):
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        dvs.add_output('alpha', val=1.5)
        self.connect('alpha', ['multi.lo_mach.alpha', 'multi.hi_mach.alpha'])

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
            'L2Convergence':1e-6,
            'L2ConvergenceCoarse':1e-2,
            'nCycles':1,
        }

        


        # cruise_analysis = om.Group()
        # cruise_analysis.add_subsystem('aero', aero, promotes=[*])

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


        scenerio_analyses = {
            'lo_mach': {'analysis' : ADflowGroup(aero_problem = ap0, 
                           solver_options = aero_options, 
                           group_options = {
                               'mesh': False,
                               'deformer': False
                           }),
                        
                        'subsystem_options' : {
                            
                        }
                        },
            'hi_mach': {'analysis' : ADflowGroup(aero_problem = ap1, 
                           solver_options = aero_options, 
                           group_options = {
                               'mesh': False,
                               'deformer': False
                           }),
                        
                        # 'subsystem_options' : {

                        # }
                        },
        }

        # the solver must be created before added the group as a subsystem.
        multi =ParallelMultipoint(scenerio_analyses = scenerio_analyses)


        self.add_subsystem('multi', multi)



################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
# om.n2(prob, show_browser=False, outfile='mphys_aero.html')

prob.run_model()

# prob.model.list_outputs()
# if MPI.COMM_WORLD.rank == 0:

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
print("Scenario 0")
print('lo cl =', prob.get_val('multi.lo_mach.cl', get_remote=True))
print('hi cl =', prob.get_val('multi.hi_mach.cl', get_remote=True))
