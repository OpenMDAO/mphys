""" Test the aero solver portion of the optimization """
import argparse
from baseclasses import AeroProblem
from adflow import ADFLOW
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpi4py import MPI
from multipoint import multiPointSparse, redirectIO

# adiabatic
# gridFile = 'meshes/visc_wall_diffbcs/plate_1e-08.cgns'
gridFile = '2dPlate_1e-02.cgns'
# gridFile = '2dPlate_1e-02_visc.cgns'


# hot
# gridFile = 'meshes/thermal_visc_wall_xrefine/plate_1e-08.cgns'

# cold
# gridFile = 'meshes/cold_thermal_visc_wall_xrefine/plate_1e-08.cgns'


# outputDir = 'solutions/' + gridFile[7:-17]
outputDir = './'

aeroOptions = {
    # 'printTiming': False,

    # Common Parameters
    'gridFile': '2dPlate_1e-02.cgns',
    'outputDirectory': './',
    # 'discretization': 'upwind',

    # 'oversetupdatemode': 'full',
    'volumevariables': ['temp'],
    'surfacevariables': ['cf', 'vx', 'vy', 'vz', 'temp', 'heattransfercoef', 'heatflux'],
    'monitorVariables':	['resturb', 'yplus'],
    # Physics Parameters
    # 'equationType': 'laminar NS',
    'equationType': 'rans',
    # 'vis2':0.0,
    'liftIndex': 2,
    'CFL': 1.0,
    # 'smoother': 'DADI',
    # 'smoother': 'runge',

    'useANKSolver': True,
    'ANKswitchtol': 10e0,
    # 'ankcfllimit': 5e6,
    # 'anksecondordswitchtol': 5e-6,
    'ankcoupledswitchtol': 5e-6,
    # NK parameters
    'useNKSolver': False,
    'nkswitchtol': 1e-8,
    
    'rkreset': False,
    'nrkreset': 40,
    'MGCycle': 'sg',
    # 'MGStart': -1,
    # Convergence Parameters
    'L2Convergence': 1e-12,
    'nCycles': 100000,
    'nCyclesCoarse': 250,
    'ankcfllimit': 5e3,
    'nsubiterturb': 5,
    'ankphysicallstolturb': 0.99,
    'anknsubiterturb': 5,
    # 'ankuseturbdadi': False,
    'ankturbkspdebug': True,

    'storerindlayer': True,
    # Turbulence model
    'eddyvisinfratio': .210438,
    'useft2SA': False,
    'turbulenceproduction': 'vorticity',
    'useblockettes': False,

}


# if MPI.COMM_WORLD.rank == 0:
#     # if gcomm.rank == 0:
#     print('redirecting output to', outputDir)
#     logFile = open(outputDir + '/log' + gridFile[-10:-5] + '.txt', 'w+b')
#     redirectIO(logFile)
    # print logFile.name + ' created sucessfully'

# atmospheric conditions
temp_air = 273  # kelvin
Pr = 0.72
mu = 1.81e-5  # kg/(m * s)

u_inf = 68  # m/s\
p_inf = 101e3


rho_inf = p_inf/(287*temp_air)

ap = AeroProblem(name='fc_'+gridFile[-10:-5], V=u_inf, T=temp_air,
                 rho=1.225, areaRef=1.0, chordRef=1.0, alpha=0, beta=0,  evalFuncs=['cl', 'cd'])


# import ipdb; ipdb.set_trace()
CFDSolver = ADFLOW(options=aeroOptions)


# this line doesn't have an effect as the wall temperature is reset later in adflow.__call__ .
#  resolving this is on the todo list
# CFDSolver.setWallTemperature(np.ones(pts.shape[0])*100.0)
# print('===============================')

# CFDSolver.setAeroProblem(ap)
# quit()
CFDSolver(ap)
# funcs = {}
# CFDSolver.evalFunctions(ap, funcs)
# print(funcs)
# k_air = 0.02620  # W/(m*K)
# pts = CFDSolver.getSurfaceCoordinates(groupName='wall')
# fluxes = CFDSolver.getHeatFluxes(groupName='wall')
# temps = CFDSolver.getWallTemperature(groupName='wall')
