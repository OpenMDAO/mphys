#rst Imports
from __future__ import print_function
import numpy
from adflow import ADFLOW
from baseclasses import *
from mpi4py import MPI
from omfsi.adflow_component import *

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce

use_openmdao = False

#rst ADflow options
aeroOptions = {
    # I/O Parameters
    'gridFile':'wing_vol.cgns',
    'outputDirectory':'.',
    'monitorvariables':['resrho','cl','cd'],
    'writeTecplotSurfaceSolution':True,

    # Physics Parameters
    'equationType':'RANS',

    # Solver Parameters
    'smoother':'dadi',
    'CFL':1.5,
    'CFLCoarse':1.25,
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

#rst Start ADflow
# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

# Add features
CFDSolver.addLiftDistribution(150, 'z')
CFDSolver.addSlices('z', numpy.linspace(0.1, 14, 10))

#rst Create AeroProblem
ap = AeroProblem(name='wing',
    mach=0.8,
    altitude=10000,
    alpha=1.5,
    areaRef=45.5,
    chordRef=3.25,
    evalFuncs=['cl','cd']
)

if use_openmdao:
    # Adflow components set up
    mesh_comp   = AdflowMesh(ap=ap,solver=CFDSolver,options=aeroOptions)
    warp_comp   = AdflowWarper(ap=ap,solver=CFDSolver)
    solver_comp = AdflowSolver(ap=ap,solver=CFDSolver)
    func_comp   = AdflowFunctions(ap=ap,solver=CFDSolver)


    # OpenMDAO set up
    prob = Problem()
    model = prob.model

    model.add_subsystem('mesh',mesh_comp)
    model.add_subsystem('warp',warp_comp)
    model.add_subsystem('solver',solver_comp)
    model.add_subsystem('funcs',func_comp)
    model.connect('mesh.x_a0',['warp.x_a'])
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
    #rst Run ADflow
    # Solve
    CFDSolver(ap)
    #rst Evaluate and print
    funcs = {}
    CFDSolver.evalFunctions(ap, funcs)
    # Print the evaluated functions
    if MPI.COMM_WORLD.rank == 0:
        print(funcs)
