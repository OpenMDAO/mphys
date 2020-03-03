#rst Imports
from __future__ import print_function
import numpy
from adflow import ADFLOW
from idwarp import USMesh
from baseclasses import *
from mpi4py import MPI
from omfsi.adflow_component import *

import openmdao.api as om

#rst ADflow options
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
    'MGCycle':'3w',
    'MGStartLevel':-1,
    'nCyclesCoarse':250,

    # ANK Solver Parameters
    'useANKSolver':True,
    'ankswitchtol':1e-1,

    # NK Solver Parameters
    'useNKSolver':True,
    'nkswitchtol':1e-4,

    # Termination Criteria
    'L2Convergence':1e-12,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':1000,
}


class AdflowSimpleAssembler(object):
    solver = None
    def __init__(self,options, ap):
        self.options = options
        self.ap = ap

        
    def get_solver(self,comm):
        solver = type(self).solver
        
        if solver is None:
            solver = ADFLOW(comm=comm, options=self.options)
            self.mesh = USMesh(comm=comm,options=self.options)
            solver.setMesh(self.mesh)
        return solver

class AeroSolver(om.Group):
    def initialize(self):
        self.options.declare('assembler')
    
    def setup(self):
        assembler = self.options['assembler']
        # Adflow components set up
        mesh_comp   = AdflowMesh(get_solver=assembler.get_solver)
        warp_comp   = AdflowWarper(get_solver=assembler.get_solver)
        solver_comp = AdflowSolver(get_solver=assembler.get_solver)
        func_comp   = AdflowFunctions(get_solver=assembler.get_solver)

        self.add_subsystem('mesh',mesh_comp)
        self.add_subsystem('warp',warp_comp)
        self.add_subsystem('solver',solver_comp)
        self.add_subsystem('funcs',func_comp)
        self.connect('mesh.x_a0_mesh',['warp.x_a'])
        self.connect('warp.x_g',['solver.x_g','funcs.x_g'])
        self.connect('solver.q',['funcs.q'])

        
        
class Top(om.Group):

    def setup(self):
        asmb1 = AdflowSimpleAssembler()
        self.add_subsystem('s1', AeroSolver(assembler=asmb1))
    
    def configure(self):
        # this was done so I could set ap after the solvers were instantiated
        # any solver specific calls would have to go here, right now

        ap1 = AeroProblem(name='wing',
                          mach=0.8,
                          altitude=10000,
                          alpha=1.5,
                          areaRef=45.5,
                          chordRef=3.25,
                          evalFuncs=['cl','cd']
        )


        self.s1.set_ap(ap1)
        #self.s1.mesh.ap = ap1
        #self.s1.warp.ap = ap1
        #self.s1.solver._set_ap(ap1)
        #self.s1.func_comp.ap = ap1





# OpenMDAO set up
prob = om.Problem()
prob.model = AeroSolver(assembler=assembler1)


prob.setup()
prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print('cl =',prob['funcs.cl'])
    print('cd =',prob['funcs.cd'])
