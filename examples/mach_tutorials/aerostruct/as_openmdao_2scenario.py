#rst Imports
from __future__ import print_function, division
import numpy
from adflow import ADFLOW
from baseclasses import *
from mpi4py import MPI

from omfsi.fsi_assembler import *
from omfsi.adflow_component import *
from omfsi.tacs_component import *
from omfsi.meld_xfer_component import *

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce
from openmdao.api import NonlinearBlockGS, LinearBlockGS

from tacs import elements, constitutive

comm = MPI.COMM_WORLD

#ADflow options
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
    'L2Convergence':1e-14,
    'L2ConvergenceCoarse':1e-2,
    'nCycles':10000,

    # force integration
    'forcesAsTractions':False,
}

# Create AeroProblem
ap = AeroProblem(name='wing',
    mach=0.8,
    altitude=10000,
    alpha=1.5,
    areaRef=45.5,
    chordRef=3.25,
    evalFuncs=['lift','drag']
)

ap.addDV('alpha',value=1.5,name='alpha')
ap.addDV('mach',value=0.8,name='mach')

aero_assembler = AdflowAssembler(comm,aero_options,ap)
aero_nnodes    = aero_assembler.solver_dict['nnodes']

aero_assembler.solver.addLiftDistribution(150, 'z')
aero_assembler.solver.addSlices('z', numpy.linspace(0.1, 14, 10))

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

func_list = ['ks_failure','mass']
tacs_setup = {'add_elements': add_elements,
              'nprocs'      : 4,
              'mesh_file'   : 'wingbox.bdf',
              'func_list'   : func_list}

struct_assembler = TacsOmfsiAssembler(comm,tacs_setup,add_elements)
struct_nnodes = struct_assembler.solver_dict['nnodes']
struct_ndof   = struct_assembler.solver_dict['ndof']

################################################################################
# Transfer scheme setup
################################################################################
meld_options = {'isym': 2,
                'n': 200,
                'beta': 0.5}

xfer_assembler = MeldAssembler(meld_options,struct_assembler,aero_assembler)


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
indeps.add_output('dv_struct',np.array(810*[0.01]))
indeps.add_output('alpha',np.array(1.5))
indeps.add_output('mach1',np.array(0.8))
indeps.add_output('mach2',np.array(0.7))
model.add_subsystem('dv',indeps)


assembler.connection_srcs['dv_struct'] = 'dv.dv_struct'

assembler.add_model_components(model)

scenario = model.add_subsystem('cruise1',Group())
scenario.nonlinear_solver = NonlinearRunOnce()
scenario.linear_solver = LinearRunOnce()

assembler.connection_srcs['dv_struct'] = 'dv.dv_struct'
assembler.connection_srcs['alpha'] = 'dv.alpha'
assembler.connection_srcs['mach'] = 'dv.mach1'

fsi_group = assembler.add_fsi_subsystem(model,scenario)
fsi_group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
fsi_group.linear_solver = LinearBlockGS(maxiter=100)

assembler.connection_srcs['dv_struct'] = 'dv.dv_struct'
assembler.connection_srcs['alpha'] = 'dv.alpha'
assembler.connection_srcs['mach'] = 'dv.mach2'

scenario2 = model.add_subsystem('cruise2',Group())
scenario2.nonlinear_solver = NonlinearRunOnce()
scenario2.linear_solver = LinearRunOnce()

fsi_group = assembler.add_fsi_subsystem(model,scenario2)
fsi_group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
fsi_group.linear_solver = LinearBlockGS(maxiter=100)

prob.setup()
prob.run_model()
