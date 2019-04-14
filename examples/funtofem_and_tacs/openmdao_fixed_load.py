"""
Mass minimization of uCRM wingbox subject to a constant vertical force

"""
from __future__ import division, print_function
import numpy as np

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce
from openmdao.api import SqliteRecorder

from tacs import elements, constitutive
from assemble_dummy_aero import FsiComps
from dummy_aero_fixed import *

################################################################################
# Tacs solver pieces
################################################################################
def add_elements(mesh):
    rho = 2500.0  # density, kg/m^3
    E = 70.0e9 # elastic modulus, Pa
    nu = 0.3 # poisson's ratio
    kcorr = 5.0 / 6.0 # shear correction factor
    ys = 350e6  # yield stress, Pa
    t= 0.010
    min_thickness = 0.00
    max_thickness = 1.00

    num_components = mesh.getNumComponents()
    for i in range(num_components):
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, t, i,
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
              'mesh_file'   : 'debug.bdf',
              'func_list'   : func_list}
meld_setup = {'isym': 1,
              'n': 200,
              'beta': 0.5}
flow_setup = {}

setup = {'tacs': tacs_setup, 'meld': meld_setup, 'flow': flow_setup}


################################################################################
# OpenMDAO setup
################################################################################
prob = Problem()
model = prob.model

indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array([0.1]))
indeps.add_output('dv_aero',1.0)
model.add_subsystem('dv',indeps)

fsi_comps = FsiComps(tacs_setup,meld_setup)
fsi_comps.add_tacs_mesh(model)
aero_mesh = AeroMesh(aero_mesh_setup=fsi_comps.aero_mesh_setup)
aero_deform = AeroDeformer(aero_deformer_setup=fsi_comps.aero_deformer_setup)
aero_solver = AeroSolver(aero_solver_setup=fsi_comps.aero_solver_setup)
aero_forces = AeroForceIntegrator(aero_force_integrator_setup=fsi_comps.aero_force_integrator_setup)

model.add_subsystem('aero_mesh',aero_mesh)
aero_group = Group()
aero_group.add_subsystem('deformer',aero_deform)
aero_group.add_subsystem('solver',aero_solver)
aero_group.add_subsystem('forces',aero_forces)
aero_group.nonlinear_solver = NonlinearRunOnce()
aero_group.linear_solver = LinearRunOnce()

fsi_comps.add_fsi_subsystems(model,aero_group,aero_nnodes=2)
fsi_comps.add_tacs_functions(model)

model.connect('aero_mesh.x_a0',['fsi_solver.disp_xfer.x_a0',
                                'fsi_solver.geo_disps.x_a0',
                                'fsi_solver.load_xfer.x_a0'])
model.connect('struct_mesh.x_s0',['fsi_solver.disp_xfer.x_s0',
                                  'fsi_solver.load_xfer.x_s0',
                                  'fsi_solver.struct.x_s0',
                                  'struct_funcs.x_s0'])
model.connect('dv.dv_struct',['fsi_solver.struct.dv_struct',
                              'struct_funcs.dv_struct'])
model.connect('dv.dv_aero',['fsi_solver.aero.solver.dv_aero'])


model.connect('fsi_solver.struct.u_s','struct_funcs.u_s')
fsi_comps.create_fsi_connections(model,nonlinear_xfer=True)

prob.setup(force_alloc_complex=True)

prob.run_model()
#prob.check_partials()
prob.check_partials(step=1e-30,compact_print=False,method='cs')
