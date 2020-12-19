# Import numpy
import numpy as np

# Import MPI
from mpi4py import MPI

# Import TACS and assorted repositories
from tacs import TACS, elements, constitutive, functions

# Set the MPI communicator
comm = MPI.COMM_WORLD

# Create the stiffness object
props = constitutive.MaterialProperties(rho=2570.0, E=70e9, nu=0.3, ys=350e6)
stiff = constitutive.SolidConstitutive(props)

# Set up the basis function
model = elements.HeatConduction3D(stiff)
basis = elements.LinearHexaBasis()
element = elements.Element3D(model, basis)

# Load structural mesh from BDF file
mesh = TACS.MeshLoader(comm)
mesh.scanBDFFile('n0012_hexa_test.bdf')

# Loop over components, creating stiffness and element object for each
num_components = mesh.getNumComponents()
for i in range(num_components):
    descriptor = mesh.getElementDescript(i)
    print('Setting element with description %s'%(descriptor))
    mesh.setElement(i, element)

assembler = mesh.createTACS(model.getVarsPerNode())

# Create the forces
forces = assembler.createVec()
forces.getArray()[:] = 1.0
assembler.applyBCs(forces)

# Set up and solve the analysis problem
res = assembler.createVec()
ans = assembler.createVec()
u = assembler.createVec()
mat = assembler.createSchurMat()
pc = TACS.Pc(mat)
subspace = 100
restarts = 2
gmres = TACS.KSM(mat, pc, subspace, restarts)

# Assemble the Jacobian and factor
alpha = 1.0
beta = 0.0
gamma = 0.0
assembler.zeroVariables()
assembler.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

# Solve the linear system
gmres.solve(forces, ans)
assembler.setVariables(ans)

# Set the element flag
flag = (TACS.OUTPUT_CONNECTIVITY |
        TACS.OUTPUT_NODES |
        TACS.OUTPUT_DISPLACEMENTS |
        TACS.OUTPUT_STRAINS)
f5 = TACS.ToFH5(assembler, TACS.SOLID_ELEMENT, flag)
f5.writeToFile('hexa.f5')
