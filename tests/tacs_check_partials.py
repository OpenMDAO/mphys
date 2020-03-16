# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp
from omfsi.tacs_component import TacsOmfsiAssembler

from tacs import elements, constitutive, functions

def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness= 0.003
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
def forcer(x_s0,ndof):
    return np.random.rand(int(x_s0.size/3*ndof))

tacs_setup = {'add_elements': add_elements,
              'get_funcs'   : get_funcs,
              'mesh_file'   : 'debug.bdf',
              'forcer_func' : forcer}

assembler = TacsOmfsiAssembler(tacs_setup,check_partials=True,add_forcer=True)

connection_srcs = {}
prob = Problem()
model = prob.model

indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array([0.01]))
model.add_subsystem('indeps',indeps)
connection_srcs['dv_struct'] = 'indeps.dv_struct'


assembler.add_model_components(model,connection_srcs)

scenario = model.add_subsystem('cruise1',Group())
fsi_group = scenario.add_subsystem('fsi_group',Group())

assembler.add_scenario_components(model,scenario,connection_srcs)

assembler.add_fsi_components(model,scenario,fsi_group,connection_srcs)
assembler.connect_inputs(model,scenario,fsi_group,connection_srcs)

prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs',compact_print=True)
