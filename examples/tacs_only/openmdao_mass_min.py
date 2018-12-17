"""
Mass minimization of uCRM wingbox subject to a constant vertical force

"""
import numpy as np

from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp, IndepVarComp

from tacs import elements, constitutive
from assemble_tacs_only import TacsComps

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
    for i in xrange(num_components):
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

def load_function(x_s,ndof):
    f_s = np.zeros(x_s.size/3*ndof)
    f_s[2::ndof] = 100.0
    return f_s

func_list = ['ks_failure','mass']
#func_list = ['ks_failure']
tacs_setup = {'add_elements': add_elements,
              'nprocs'      : 4,
              'mesh_file'   : 'CRM_box_2nd.bdf',
              'func_list'   : func_list}

setup = {'tacs': tacs_setup}

tacs_comps = TacsComps()

################################################################################
# OpenMDAO setup
################################################################################
prob = Problem()
model = prob.model

indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array(240*[0.005]))
model.add_subsystem('indeps',indeps,promotes=['dv_struct'])

tacs_comps.add_tacs_subsystems(model,setup,load_function=load_function)


prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

model.add_design_var('dv_struct',lower=0.001,upper=0.075,scaler=100.0)

model.add_objective('mass',scaler=1/1000.0/100.0)
model.add_constraint('f_struct',upper = 2.0/3.0,scaler=1.0)

prob.setup()

#prob.run_model()
prob.run_driver()
#prob.check_partials(compact_print=True)
