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

class Distributor(ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('dv_s')
        self.add_output('dv_struct',shape=810)
    def compute(self,inputs,outputs):
        outputs['dv_struct'][:] = inputs['dv_s']

#ADflow options
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
    'nCycles':500,

    # force integration
    'forcesAsTractions':False,
}

# Create aero solver
CFDSolver = ADFLOW(options=aeroOptions)

# Add features
CFDSolver.addLiftDistribution(150, 'z')
CFDSolver.addSlices('z', numpy.linspace(0.1, 14, 10))

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

################################################################################
# TACS setup
################################################################################
def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness= 0.010
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

func_list = ['ks_failure']
tacs_setup = {'add_elements': add_elements,
              'nprocs'      : 4,
              'mesh_file'   : 'wingbox.bdf',
              'func_list'   : func_list}

################################################################################
# Transfer scheme setup
################################################################################
meld_setup = {'isym': 2,
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

aero_nnodes = int(CFDSolver.getSurfaceCoordinates().size /3)


# OpenMDAO problem set up
prob = Problem()
model = prob.model

model.nonlinear_solver = NonlinearRunOnce()
model.linear_solver = LinearRunOnce()

class Summer(ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('dv_struct',shape=810)
        self.add_output('mass')
    def compute(self,inputs,outputs):
        outputs['mass'] = sum(inputs['dv_struct'])
    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            d_outputs['mass'] += sum(d_inputs['dv_struct'])
        if mode == 'rev':
            d_inputs['dv_struct'] += d_outputs['mass']

#Add the components and groups to the model
indeps = IndepVarComp()
indeps.add_output('alpha',np.array(1.5))
indeps.add_output('dv_struct',np.array(810*[0.005]))
model.add_subsystem('dvs',indeps)

assembler = FsiComps(tacs_setup,meld_setup)
assembler.add_tacs_mesh(model)
model.add_subsystem('aero_mesh',aero_mesh)

assembler.add_fsi_subsystems(model,aero_group,aero_nnodes)

assembler.add_tacs_functions(model)
model.add_subsystem('aero_funcs',aero_funcs)
model.add_subsystem('summer',Summer())

# Connect the components
model.connect('aero_mesh.x_a0',['fsi_solver.disp_xfer.x_a0',
                                'fsi_solver.geo_disps.x_a0',
                                'fsi_solver.load_xfer.x_a0'])
model.connect('struct_mesh.x_s0',['fsi_solver.disp_xfer.x_s0',
                                  'fsi_solver.load_xfer.x_s0',
                                  'fsi_solver.struct.x_s0',
                                  'struct_funcs.x_s0',
                                  'struct_mass.x_s0'])

model.connect('dv.dv_struct',['fsi_solver.struct.dv_struct',
                              'struct_funcs.dv_struct',
                              'struct_mass.dv_struct'])
model.connect('dvs.alpha',['fsi_solver.aero.solver.alpha',
                           'fsi_solver.aero.forces.alpha',
                           'aero_funcs.alpha'])

model.connect('fsi_solver.aero.deformer.x_g','aero_funcs.x_g')
model.connect('fsi_solver.aero.solver.q','aero_funcs.q')

model.connect('fsi_solver.struct.u_s','struct_funcs.u_s')
assembler.create_fsi_connections(model,nonlinear_xfer=True)

model.add_subsystem('trim',ExecComp('balance = lift - 9.81*3000*mass'),promotes=['balance'])
model.connect('summer.mass','trim.mass')
model.connect('aero_funcs.lift','trim.lift')

prob.driver = ScipyOptimizeDriver(debug_print=['objs','nl_cons'],maxiter=1500)
prob.driver.options['optimizer'] = 'SLSQP'

model.add_design_var('dvs.dv_struct',lower=0.001,upper=0.075,scaler=1000.0/1.0)
model.add_design_var('dvs.alpha',lower=-5.0,upper=5.0,scaler=1.0/1.0)

model.add_objective('summer.mass',scaler=1.0/100000.0)
model.add_constraint('struct_funcs.f_struct',lower = 0.0, upper = 2.0/3.0,scaler=1000.0/1.0)
model.add_constraint('balance',equals = 0.0, scaler=1.0/1000.0)

prob.setup()
prob.run_driver()

for i in range(810):
    print('final dvs',i,prob['dvs.dv_struct'][i])
print('final alpha',i,prob['dvs.alpha'])
