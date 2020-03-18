#rst Imports
from __future__ import print_function
import numpy as np
from adflow import ADFLOW
from baseclasses import *
from mpi4py import MPI

from omfsi import FsiAssembler, GeoDispAssembler, GeoDisp
from omfsi import AdflowAssembler, AdflowMesh, AdflowWarper, AdflowSolver, AdflowFunctions
from omfsi import TacsOmfsiAssembler, functions, TACS
from omfsi import RLTAssembler, RLTDisplacementTransfer

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce
from openmdao.api import NonlinearBlockGS, LinearBlockGS

from tacs import elements, constitutive

comm = MPI.COMM_WORLD

#ADflow options
aero_options = {
    # I/O Parameters
    'gridFile':'debug.cgns',
    'outputDirectory':'.',
    'monitorvariables':['resrho','cl','cd'],
    'writeTecplotSurfaceSolution':False,
    'writeVolumeSolution':False,
    'writeSurfaceSolution':False,
    'printiterations':False,
    'printtiming':False,
    'printwarnings':False,

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
    'L2convergencerel':1e-1,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':200,

    # force integration
    'forcesAsTractions':False,
}

# Create AeroProblem
ap = AeroProblem(name='debug',
    mach=0.3,
    altitude=10000,
    alpha=1.5,
    areaRef=16.0*32.0,
    chordRef=16.0,
    evalFuncs=['lift','drag']
)

ap.addDV('alpha',value=1.5,name='alpha')
ap.addDV('mach',value=0.3,name='mach')

aero_assembler = AdflowAssembler(aero_options,ap)

################################################################################
# TACS setup
################################################################################
def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness= 0.50
    min_thickness = 0.002
    max_thickness = 0.05

    num_components = mesh.getNumComponents()
    for i in range(num_components):
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

def get_funcs(tacs):
    ks_weight = 50.0
    return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

def f5_writer(tacs):
    flag = (TACS.ToFH5.NODES |
            TACS.ToFH5.DISPLACEMENTS |
            TACS.ToFH5.STRAINS |
            TACS.ToFH5.EXTRAS)
    f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
    f5.writeToFile('debug.f5')

tacs_setup = {'add_elements': add_elements,
              'mesh_file'   : 'debug.bdf',
              'get_funcs'   : get_funcs,
              'f5_writer'   : f5_writer}

struct_assembler = TacsOmfsiAssembler(tacs_setup)

################################################################################
# Transfer scheme setup
################################################################################
transfer_options = {
    'transfergaussorder': 2,
}

xfer_assembler = RLTAssembler(transfer_options,struct_assembler,aero_assembler)

################################################################################
# OpenMDAO setup
################################################################################

assembler = FsiAssembler(struct_assembler,aero_assembler,xfer_assembler)

# OpenMDAO problem set up
prob = Problem()
model = prob.model
model.nonlinear_solver = NonlinearRunOnce()
model.linear_solver = LinearRunOnce()


#Add the components and groups to the model
indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array(1*[2.0]))
indeps.add_output('alpha',np.array(1.5))
indeps.add_output('mach',np.array(0.3))
model.add_subsystem('dv',indeps)

assembler.add_model_components(model)

scenario = model.add_subsystem('cruise1',Group())
scenario.nonlinear_solver = NonlinearRunOnce()
scenario.linear_solver = LinearRunOnce()

# Connect the components
assembler.connection_srcs['dv_struct'] = 'dv.dv_struct'
assembler.connection_srcs['alpha'] = 'dv.alpha'
assembler.connection_srcs['mach'] = 'dv.mach'
fsi_group = assembler.add_fsi_subsystem(model,scenario)
fsi_group.nonlinear_solver = NonlinearBlockGS(maxiter=100, iprint=2)
fsi_group.linear_solver = LinearBlockGS(maxiter=100)


prob.setup()
prob.run_model()
prob.check_partials(step=1e-8, compact_print=True)

if MPI.COMM_WORLD.rank == 0:
    print('lift =',prob[scenario.name + '.aero_funcs.lift'])
    print('drag =',prob[scenario.name + '.aero_funcs.drag'])
