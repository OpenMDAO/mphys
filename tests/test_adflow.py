""" tests the componets included in the MPHYS wrapper for TACS"""
import unittest
import openmdao.api as om
import numpy as np
from mphys.mphys_adflow import *


@unittest.skip('')
class TestAdflowSubsys(unittest.TestCase):
    N_Procs = 2

    def setUp(self):
        """ keep the options used to init the adflow solver options here """
 
        
        
        self.solver_options = {
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
            'nCycles':1,
        }

        self.ap = AeroProblem(
            name='test',
            mach=0.8,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['cl','cd']
        )
        self.ap.addDV('alpha',value=1.5,name='alpha')
        
        self.top =  om.Group()

    def test_mesh(self):
        self.top.add_subsystem('mesh',AdflowMesh(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

    def test_geo_disp(self):
        self.top.add_subsystem('geo_disp', Geo_Disp(solver_options=self.solver_options))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

    def test_solver(self):
        self.top.add_subsystem('solver',AdflowSolver(solver_options=self.solver_options,
                                                     aero_problem=self.ap))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()
    
    def test_force(self):
        self.top.add_subsystem('forces',AdflowForces(solver_options=self.solver_options,
                                                     aero_problem=self.ap))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()

    def test_functions(self):
        self.top.add_subsystem('funcs',AdflowFunctions(solver_options=self.solver_options,
                                                     aero_problem=self.ap))
        
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        prob.run_model()



class TestAdflowGroup(unittest.TestCase):
    N_Procs = 2

    def setUp(self):
        """ keep the options used to init the tacs solver options here """

        
        self.solver_options = {
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
            'nCycles':1,
        }

        self.ap = AeroProblem(
            name='test',
            mach=0.8,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['cl','cd']
        )
        self.ap.addDV('alpha',value=1.5,name='alpha')
        
        self.top =  om.Group()

    def test_group(self):

        self.top.add_subsystem('aero', ADflowGroup(aero_problem = self.ap, 
                                            solver_options = self.solver_options, 
                                            group_options = {
                                                'mesh': True,
                                                'geo_disp':False,
                                                'deformer': True
                                            }))
        prob = om.Problem()

        prob.model = self.top
        prob.setup()
        om.n2(prob, show_browser=False, outfile='mphys_test_adflow_group.html')

        prob.run_model()


if __name__ == '__main__':
    unittest.main()