"""
Mass minimization of uCRM wingbox subject to a constant vertical force

"""
from __future__ import division, print_function
import numpy as np

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import IndepVarComp, Group
from openmdao.api import NonlinearBlockGS, LinearBlockGS

from tacs import elements, constitutive, TACS
from omfsi.tacs_component import *

################################################################################
# Tacs solver pieces
################################################################################
def add_elements(mesh):
    rho = 2500.0  # density, kg/m^3
    E = 70.0e9 # elastic modulus, Pa
    nu = 0.3 # poisson's ratio
    kcorr = 5.0 / 6.0 # shear correction factor
    ys = 350e6  # yield stress, Pa
    thickness = 0.020
    min_thickness = 0.00
    max_thickness = 1.00

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

def forcer_function(x_s,ndof):
    # apply uniform z load
    f_s = np.zeros(int(x_s.size/3)*ndof)
    f_s[2::ndof] = 100.0
    return f_s

def f5_writer(tacs):
    flag = (TACS.ToFH5.NODES |
            TACS.ToFH5.DISPLACEMENTS |
            TACS.ToFH5.STRAINS |
            TACS.ToFH5.EXTRAS)
    f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
    f5.writeToFile('ucrm.f5')

tacs_setup = {'add_elements': add_elements,
              'mesh_file'   : 'CRM_box_2nd.bdf',
              'get_funcs'   : get_funcs,
              'forcer_func' : forcer_function,
              'f5_writer'   : f5_writer}

assembler = TacsOmfsiAssembler(tacs_setup,add_forcer=True)

################################################################################
# OpenMDAO setup
################################################################################
prob = Problem()
model = prob.model

connection_srcs = {}

scenario = model.add_subsystem('uniform_load',Group())
coupling_group = scenario.add_subsystem('coupling',Group())

assembler.add_model_components(model,connection_srcs)
assembler.add_scenario_components(model,scenario,connection_srcs)
assembler.add_fsi_components(model,scenario,coupling_group,connection_srcs)

indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array(240*[0.0031]))
model.add_subsystem('indeps',indeps)
connection_srcs['dv_struct'] = 'indeps.dv_struct'

assembler.connect_inputs(model,scenario,coupling_group,connection_srcs)

model.set_order([indeps.name,'struct_mesh',scenario.name])

prob.driver = ScipyOptimizeDriver(debug_print=['objs','nl_cons'],maxiter=1500)
prob.driver.options['optimizer'] = 'SLSQP'

model.add_design_var('indeps.dv_struct',lower=0.001,upper=0.075,scaler=1.0/1.0)

model.add_objective(scenario.name+'.struct_mass.mass',scaler=1.0/100000.0)
model.add_constraint(scenario.name+'.struct_funcs.f_struct',lower = 0.0, upper = 2.0/3.0,scaler=1000.0/1.0)

coupling_group.nonlinear_solver = NonlinearBlockGS()
coupling_group.linear_solver = LinearBlockGS()

prob.setup()

prob.run_model()
#prob.run_driver()
#for i in range(240):
#    print('final dvs',i,prob['dv_struct'][i])
