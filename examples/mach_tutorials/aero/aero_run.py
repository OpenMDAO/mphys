#rst Imports
from __future__ import print_function
import numpy
from adflow import ADFLOW
from baseclasses import AeroProblem
from idwarp import USMesh
from mpi4py import MPI
from omfsi.adflow_component import AdflowAssembler, AdflowMesh, AdflowWarper, AdflowSolver

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce


use_openmdao = True

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
    'MGCycle':'2w',
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


#rst Create AeroProblem
ap = AeroProblem(name='wing',
    mach=0.8,
    altitude=10000,
    alpha=1.5,
    areaRef=45.5,
    chordRef=3.25,
    evalFuncs=['cl','cd']
)

class AdflowSimpleAssembler(object):
    def __init__(self,options):
        self.options = options
        self.solver = None
    def get_solver(self,comm):
        if self.solver is None:
            self.solver = ADFLOW(options=self.options)
            self.mesh = USMesh(comm=comm,options=self.options)
            self.solver.setMesh(self.mesh)
        return self.solver

if use_openmdao:
    assembler = AdflowSimpleAssembler(aero_options)
    # Adflow components set up
    mesh_comp   = AdflowMesh(ap=ap,get_solver=assembler.get_solver)
    warp_comp   = AdflowWarper(ap=ap,get_solver=assembler.get_solver)
    solver_comp = AdflowSolver(ap=ap,get_solver=assembler.get_solver)
    func_comp   = AdflowFunctions(ap=ap,get_solver=assembler.get_solver)

    # OpenMDAO set up
    prob = Problem()
    model = prob.model

    model.add_subsystem('mesh',mesh_comp)
    model.add_subsystem('warp',warp_comp)
    model.add_subsystem('solver',solver_comp)
    model.add_subsystem('funcs',func_comp)
    model.connect('mesh.x_a0_mesh',['warp.x_a'])
    model.connect('warp.x_g',['solver.x_g','funcs.x_g'])
    model.connect('solver.q',['funcs.q'])

    model.nonlinear_solver = NonlinearRunOnce()
    model.linear_solver = LinearRunOnce()

    prob.setup()
    prob.run_model()

    if MPI.COMM_WORLD.rank == 0:
        print('cl =',prob['funcs.cl'])
        print('cd =',prob['funcs.cd'])

else:
    CFDSolver = ADFLOW(options=aero_options)
    #rst Run ADflow
    # Solve
    CFDSolver(ap)
    #rst Evaluate and print
    funcs = {}
    CFDSolver.evalFunctions(ap, funcs)
    # Print the evaluated functions
    if MPI.COMM_WORLD.rank == 0:
        print(funcs)
