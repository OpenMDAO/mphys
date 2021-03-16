# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np

import openmdao.api as om
from mphys.mphys_tacs import TacsBuilder
from mphys.multipoint import Multipoint
from mphys.scenario_structural import ScenarioStructural

from tacs import elements, constitutive, functions


def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness = 0.003
    min_thickness = 0.002
    max_thickness = 0.05

    num_components = mesh.getNumComponents()
    for i in range(num_components):
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                     min_thickness, max_thickness)
        element = None
        if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.MITCShell(2, stiff, component_num=i)
        mesh.setElement(i, element)
    ndof = 6
    ndv = num_components

    return ndof, ndv


def get_funcs(tacs):
    ks_weight = 50.0
    return [functions.KSFailure(tacs, ks_weight), functions.StructuralMass(tacs)]


def forcer(x_s0, ndof):
    return np.random.rand(int(x_s0.size/3*ndof))


class Top(Multipoint):

    def setup(self):

        tacs_options = {'add_elements': add_elements,
                        'get_funcs': get_funcs,
                        'mesh_file': '../input_files/debug.bdf',
                        'load_function': forcer}

        tacs_builder = TacsBuilder(tacs_options, check_partials=True)
        tacs_builder.initialize(self.comm)
        ndv_struct = tacs_builder.get_ndv()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))

        self.add_subsystem('mesh', tacs_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=tacs_builder))
        self.connect('mesh.x_struct0', 'analysis.x_struct0')
        self.connect('dv_struct', 'analysis.dv_struct')


prob = om.Problem()
prob.model = Top()

prob.setup(mode='rev', force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs', compact_print=True)
