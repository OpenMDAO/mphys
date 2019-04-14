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
from omfsi.assemble_tacs_only import TacsComps

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

def load_function(x_s,ndof):
    f_s = np.zeros(int(x_s.size/3)*ndof)
    f_s[2::ndof] = 100.0
    return f_s

func_list = ['mass','compliance']
tacs_setup = {'add_elements': add_elements,
              'nprocs'      : 4,
              'mesh_file'   : 'debug.bdf',
              'func_list'   : func_list}

setup = {'tacs': tacs_setup}

tacs_comps = TacsComps()

################################################################################
# OpenMDAO setup
################################################################################
prob = Problem()
model = prob.model

indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array(1*[0.2]))
model.add_subsystem('indeps',indeps,promotes=['dv_struct'])

tacs_comps.add_tacs_subsystems(model,setup,load_function=load_function)

prob.setup(force_alloc_complex=True)

prob.run_model()
#prob.check_partials(step=1e-30,method='cs')
#prob.check_partials(step=1e-30,compact_print=True,method='cs')
prob.check_partials(step=1e-30,compact_print=False,method='cs')
