from __future__ import print_function
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys
# Import SU2
import pysu2
from tacs import TACS, elements, constitutive
from funtofem import TransferScheme

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# initialize SU2
SU2_CFD_ConfigFile = 'sz_flatplate_flow.cfg'
SU2Driver = pysu2.CSinglezoneDriver(SU2_CFD_ConfigFile, 1, comm)

# initialize markers
CHTMarkerID = None
CHTMarker = 'wall'

# Get all the tags with the CHT option
# if this list is empty, you probably didn't add
# MARKER_PYTHON_CUSTOM= ( your_marker ) to the .cfg file
CHTMarkerList =  SU2Driver.GetAllCHTMarkersTag()

# Get all the markers defined on this rank and their associated indices.
allMarkerIDs = SU2Driver.GetAllBoundaryMarkers()

#Check if the specified marker has a CHT option and if it exists on this rank.
if CHTMarker in CHTMarkerList and CHTMarker in allMarkerIDs.keys():
    CHTMarkerID = allMarkerIDs[CHTMarker]

if CHTMarkerID != None:
    nVertex_CHTMarker = SU2Driver.GetNumberVertices(CHTMarkerID)

# get aero nodes
X = []
for i in range(nVertex_CHTMarker):
    X.extend([SU2Driver.GetVertexCoordX(CHTMarkerID, i),
              SU2Driver.GetVertexCoordY(CHTMarkerID, i),
              SU2Driver.GetVertexCoordZ(CHTMarkerID, i)])

# initialize TACS
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
opp_edge = []
for i in range(len(Xpts_array) // 3):

    # check if it's on the flow edge
    if Xpts_array[3*i+1] == 0.0:
        plate_surface.extend(Xpts_array[3*i:3*i+3])
        mapping.append(i)

plate_surface = np.array(plate_surface)
X = np.array(X)

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
normal_flux = np.zeros(nVertex_CHTMarker)
theta = np.zeros(nVertex_CHTMarker)
temp_check = np.zeros(nVertex_CHTMarker)
res_holder = np.zeros(len(mapping))
ans_holder = np.zeros(len(mapping))

# this iterates over a fixed number of iterations
# this could just be a criteria based on how much the temperature changes between
# subsequent iterations, similar to the conduction examples
# however, then there would be no guarantee that the CFD has converged to
# the desired tolerance. so instead I'm just using a fixed number of iterations
Iter = 0
MaxIter = 3000
while (Iter < MaxIter):
    # Time iteration preprocessing
    SU2Driver.Preprocess(Iter)
    SU2Driver.BoundaryConditionsUpdate()

    # get the normal heat flux from su2
    for iVertex in range(nVertex_CHTMarker):
        # if this line breaks the code, need to add the GetVertexAreaHeatFlux function
        # to python wrapper and CDriver.
        normal_flux[iVertex] = SU2Driver.GetVertexAreaHeatFlux(CHTMarkerID, iVertex)
        temp_check[iVertex] = SU2Driver.GetVertexTemperature(CHTMarkerID, iVertex)
    print('avg temp at wall = ', np.mean(np.array(temp_check)))
    res.zeroEntries()
    res_array = res.getArray()

    # set flux into TACS
    meld.transferFlux(normal_flux, res_holder)
    
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
    
    # set theta temps into SU2 for next run
    # don't do this on iter=0 because it'll set all zeros in
    if Iter > 0:
        for iVertex in range(nVertex_CHTMarker):
            setTemp = temp_check[iVertex] + 0.5*(theta[iVertex] - temp_check[iVertex])
            SU2Driver.SetVertexTemperature(CHTMarkerID, iVertex, setTemp)

    # Run an iteration of the CFD
    SU2Driver.Run()
    SU2Driver.Monitor(Iter)

    SU2Driver.Output(Iter)

    print(Iter)
    Iter += 1

# Set the element flag
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(assembler, TACS.SCALAR_2D_ELEMENT, flag)
f5.writeToFile('tacs_flatplate.f5')

SU2Driver.Postprocessing()
        
