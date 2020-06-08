from __future__ import print_function
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
# Import SU2
import pysu2
from tacs import TACS, elements, constitutive
import os
import sys

from adflow import ADFLOW
from baseclasses import AeroProblem

baseDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(baseDir,'../../../../'))


from funtofem import TransferScheme

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


aeroOptions = {
    # 'printTiming': False,

    # Common Parameters
    'gridFile': '2dPlate_1e-02.cgns',
    'outputDirectory': './',
    # 'discretization': 'upwind',

    # 'oversetupdatemode': 'full',
    'volumevariables': ['temp'],
    'surfacevariables': ['cf', 'vx', 'vy', 'vz', 'temp', 'heattransfercoef', 'heatflux'],
    'monitorVariables':	['resturb', 'yplus', 'heatflux'],
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
    'nCycles': 1000,
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

# atmospheric conditions
temp_air = 273  # kelvin
Pr = 0.72
mu = 1.81e-5  # kg/(m * s)

u_inf = 68  # m/s\
p_inf = 101e3


rho_inf = p_inf/(287*temp_air)
CFDSolver = ADFLOW(options=aeroOptions)


ap = AeroProblem(name='fc_conv', V=u_inf, T=temp_air,
                 rho=1.225, areaRef=1.0, chordRef=1.0, alpha=0, beta=0,  evalFuncs=['cl', 'cd'])


group = 'wall'
BCVar = 'Temperature'

BCData = CFDSolver.getBCData(groupNames=[group])
ap.setBCVar(BCVar,  BCData[group][BCVar], group)
ap.addDV(BCVar, familyGroup=group, name='wall_temp')



X = CFDSolver.getSurfacePoints(groupName=CFDSolver.allIsothermalWallsGroup)
nAeroNodes = len(X)

## initialize TACS
# Create the constitutvie propertes and model
props = constitutive.MaterialProperties()
con = constitutive.PlaneStressConstitutive(props)
heat = elements.HeatConduction2D(con)

# Create the basis class
quad_basis = elements.LinearQuadBasis()

# Create the element
element = elements.Element2D(heat, quad_basis)

# Load in the mesh
mesh = TACS.MeshLoader(comm)
mesh.scanBDFFile('flatplate.bdf')

# Set the element
mesh.setElement(0, element)

# Create the assembler object
varsPerNode = heat.getVarsPerNode()
assembler = mesh.createTACS(varsPerNode)

# get structures nodes
Xpts = assembler.createNodeVec()
assembler.getNodes(Xpts)
Xpts_array = Xpts.getArray()

# get mapping of flow edge
plate_surface = []
mapping = []
for i in range(len(Xpts_array) // 3):

    # check if it's on the flow edge
    if Xpts_array[3*i+1] == 0.0:
        plate_surface.extend(Xpts_array[3*i:3*i+3])
        mapping.append(i)


plate_surface = np.array(plate_surface)
X = X.flatten()

# Create the vectors/matrices
res = assembler.createVec()
ans = assembler.createVec()
mat = assembler.createSchurMat()
pc = TACS.Pc(mat)

# Assemble the heat conduction matrix
assembler.assembleJacobian(1.0, 0.0, 0.0, res, mat)
pc.factor()
gmres = TACS.KSM(mat, pc, 20)

# initialize MELDThermal
meld = TransferScheme.pyMELDThermal(comm, comm, 0, comm, 0, -1, 10, 0.5) #axis of symmetry, num nearest, beta

meld.setStructNodes(plate_surface)
meld.setAeroNodes(X)
meld.initialize()

# allocate some storage arrays
normal_flux = np.zeros(nAeroNodes)
theta = np.zeros(nAeroNodes)
# temp_check = np.zeros(nAeroNodes)
res_holder = np.zeros(len(mapping))
ans_holder = np.zeros(len(mapping))

# this iterates over a fixed number of iterations
# this could just be a criteria based on how much the temperature changes between
# subsequent iterations, similar to the conduction examples
# however, then there would be no guarantee that the CFD has converged to
# the desired tolerance. so instead I'm just using a fixed number of iterations
Iter = 0
MaxIter = 10
while (Iter < MaxIter):
    # Time iteration preprocessing
    # SU2Driver.Preprocess(Iter)
    # SU2Driver.BoundaryConditionsUpdate()

    # # get the normal heat flux from su2
    # for iVertex in range(nVertex_CHTMarker):
    #     # if this line breaks the code, need to add the GetVertexAreaHeatFlux function
    #     # to python wrapper and CDriver.
    #     normal_flux[iVertex] = SU2Driver.GetVertexAreaHeatFlux(CHTMarkerID, iVertex)
    #     temp_check[iVertex] = SU2Driver.GetVertexTemperature(CHTMarkerID, iVertex)

    CFDSolver(ap)
    heat = CFDSolver.getHeatFluxes()
    print('avg normla_flux at wall = ', np.mean(np.array(heat)))
    res.zeroEntries()
    res_array = res.getArray()

    # set flux into TACS
    meld.transferFlux(np.array(heat), res_holder)
    
    # transfer flux from res holder to res array based on mapping
    # this takes the fluxes, which correspond to the upper edge of the plate
    # and places them in the residual array, which is the size of the entire plate
    # the purpose of the mapping is to place the fluxes on the nodes that correspond
    # to the upper edge of the plate
    for i in range(len(mapping)):
        res_array[mapping[i]] = res_holder[i]

    # set flux into assembler
    assembler.setBCs(res)
    # solve thermal problem
    gmres.solve(res, ans)
    assembler.setVariables(ans)
    
    ans_array = ans.getArray()
    
    # get specifically the temps from the nodes in the mapping
    # i.e. the surface nodes of the structure
    for i in range(len(mapping)):
        ans_holder[i] = ans_array[mapping[i]]
    print('avg temp to set into SU2 = ', np.mean(np.array(ans_holder)))
    
    # transfer surface temps to theta (size of nodes on aero side)
    meld.transferTemp(ans_holder, theta)
    
    ap.setDesignVars({'wall_temp_(1,1)':theta})
    print(Iter)
    Iter += 1

# Set the element flag
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(assembler, TACS.SCALAR_2D_ELEMENT, flag)
f5.writeToFile('tacs_flatplate.f5')

# SU2Driver.Postprocessing()
        
