#rst Imports
from __future__ import print_function
import numpy
from adflow import ADFLOW
from baseclasses import *
from mpi4py import MPI
from omfsi.adflow_component import *
from omfsi.assemble import *

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce

from tacs import elements, constitutive

use_openmdao = False

#ADflow options
aeroOptions = {
    # I/O Parameters
    'gridFile':'debug.cgns',
    'outputDirectory':'.',
    'monitorvariables':['resrho','cl','cd'],
    'writeTecplotSurfaceSolution':False,
    'writeVolumeSolution':False,
    'writeSurfaceSolution':False,

    # Physics Parameters
    'equationType':'euler',

    # Solver Parameters
    'smoother':'dadi',
    'CFL':1.5,
    'CFLCoarse':1.25,
    'MGCycle':'sg',
    'MGStartLevel':-1,
    #'nCyclesCoarse':250,

    # ANK Solver Parameters
    'useANKSolver':True,
    'ankswitchtol':1e-1,

    # NK Solver Parameters
    'useNKSolver':True,
    'nkswitchtol':1e-4,

    # Termination Criteria
    'L2Convergence':1e-12,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':200,

    # force integration
    'forcesAsTractions':False,
}

# Create aero solver
CFDSolver = ADFLOW(options=aeroOptions)

# Add features
#CFDSolver.addLiftDistribution(150, 'z')
#CFDSolver.addSlices('z', numpy.linspace(0.1, 14, 10))

# Create AeroProblem
ap = AeroProblem(name='debug',
    mach=0.3,
    altitude=10000,
    alpha=1.5,
    areaRef=16.0*32.0,
    chordRef=16.0,
    evalFuncs=['lift','drag']
)

################################################################################
# TACS setup
################################################################################
def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness= 0.020
    min_thickness = 0.002
    max_thickness = 0.05

    num_components = mesh.getNumComponents()
    for i in xrange(num_components):
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                     min_thickness, max_thickness)
        element = None
        if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.MITCShell(2,stiff,component_num=i)
        mesh.setElement(i, element)

    ndof = 6
    ndv = num_components

    return ndof, ndv

func_list = ['ks_failure','mass']
tacs_setup = {'add_elements': add_elements,
              'nprocs'      : 1,
              'mesh_file'   : 'debug.bdf',
              'func_list'   : func_list}

################################################################################
# Transfer scheme setup
################################################################################
meld_setup = {'isym': -1,
              'n': 200,
              'beta': 0.5}

################################################################################
# OpenMDAO setup
################################################################################
# Adflow components set up
aero_mesh   = AdflowMesh(ap=ap,solver=CFDSolver,options=aeroOptions)
aero_deform = AdflowWarper(ap=ap,solver=CFDSolver)
aero_solver = AdflowSolver(ap=ap,solver=CFDSolver)
aero_forces = AdflowForces(ap=ap,solver=CFDSolver)
aero_funcs  = AdflowFunctions(ap=ap,solver=CFDSolver)

# Aero group
aero_group = Group()
aero_group.add_subsystem('deformer',aero_deform)
aero_group.add_subsystem('solver',aero_solver)
aero_group.add_subsystem('forces',aero_forces)
aero_group.nonlinear_solver = NonlinearRunOnce()
aero_group.linear_solver = LinearRunOnce()

aero_nnodes = CFDSolver.getSurfaceCoordinates().size /3


# OpenMDAO problem set up
prob = Problem()
model = prob.model

model.nonlinear_solver = NonlinearRunOnce()
model.linear_solver = LinearRunOnce()

#Add the components and groups to the model
indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array(1*[0.01]))
model.add_subsystem('dv',indeps)

assembler = FsiComps(tacs_setup,meld_setup)
assembler.add_tacs_mesh(model)
model.add_subsystem('aero_mesh',aero_mesh)

assembler.add_fsi_subsystems(model,aero_group,aero_nnodes)

assembler.add_tacs_functions(model)
model.add_subsystem('aero_funcs',aero_funcs)

# Connect the components
model.connect('aero_mesh.x_a0',['fsi_solver.disp_xfer.x_a0',
                                'fsi_solver.geo_disps.x_a0',
                                'fsi_solver.load_xfer.x_a0'])
model.connect('struct_mesh.x_s0',['fsi_solver.disp_xfer.x_s0',
                                  'fsi_solver.load_xfer.x_s0',
                                  'fsi_solver.struct.x_s0',
                                  'struct_funcs.x_s0'])
model.connect('dv.dv_struct',['fsi_solver.struct.dv_struct',
                              'struct_funcs.dv_struct'])


model.connect('fsi_solver.aero.deformer.x_g','aero_funcs.x_g')
model.connect('fsi_solver.aero.solver.q','aero_funcs.q')

model.connect('fsi_solver.struct.u_s','struct_funcs.u_s')
assembler.create_fsi_connections(model,nonlinear_xfer=True)


prob.setup()
prob.run_model()
prob.check_partials()

if MPI.COMM_WORLD.rank == 0:
    print('cl =',prob['aero_funcs.cl'])
    print('cd =',prob['aero_funcs.cd'])
