"""
@File    :   test_aero_derivs.py
@Time    :   2020/12/17
@Author  :   Josh Anibal
@Desc    :   File for testing the derivatives of the mphys adflow wrapper
"""

# === Standard Python modules ===
import unittest
import os

# === External Python modules ===
import numpy as np

# from mpi4py import MPI

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint
from mphys.scenario_structural import ScenarioStructural

from tacs import elements, constitutive, functions, TACS
from mphys.mphys_tacs import TacsBuilder


baseDir = os.path.dirname(os.path.abspath(__file__))


# factors
ks_weight = 50.0
min_thickness = 0.0016
max_thickness = 0.02
initial_thickness = 0.012
thickness_ref = initial_thickness
load_factor = 2.5
alpha0 = 3.725


def add_elements(mesh):
    rho = 2780.0  # density, kg/m^3
    E = 73.1e9  # elastic modulus, Pa
    nu = 0.33  # poisson's ratio
    kcorr = 5.0 / 6.0  # shear correction factor
    ys = 324.0e6  # yield stress, Pa
    thickness = initial_thickness
    min_t = min_thickness
    max_t = max_thickness

    num_components = mesh.getNumComponents()
    for i in range(num_components):
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i, min_t, max_t)
        element = None
        if descript in ['CQUAD', 'CQUADR', 'CQUAD4']:
            element = elements.MITCShell(2, stiff, component_num=i)
        mesh.setElement(i, element)

    ndof = 6
    ndv = num_components

    return ndof, ndv


def get_funcs(tacs):
    ks = functions.KSFailure(tacs, ks_weight)
    ks.setLoadFactor(load_factor)
    ms = functions.StructuralMass(tacs)
    return [ks, ms]


class Top(Multipoint):
    def setup(self):

        struct_options = {
            'add_elements': add_elements,
            'get_funcs': get_funcs,
            'mesh_file': os.path.join(baseDir, '../input_files/debug.bdf'),
        }
        struct_builder = TacsBuilder(struct_options, check_partials=False)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()
        f_size = struct_builder.get_ndof() * struct_builder.get_number_of_nodes()

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc for DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct * [0.01]))
        dvs.add_output('f_struct', np.ones(f_size))

        self.add_subsystem('mesh', struct_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=struct_builder))
        self.connect('mesh.x_struct0', 'analysis.x_struct0')
        self.connect('dv_struct', 'analysis.dv_struct')
        self.connect('f_struct', 'analysis.f_struct')


class TestTACs(unittest.TestCase):
    def setUp(self):

        ################################################################################
        # OpenMDAO setup
        ################################################################################
        prob = om.Problem()
        prob.model = Top()

        # DVs
        #prob.model.add_design_var('dv_struct', indices=[0], lower=-5, upper=10, ref=10.0)
        #prob.model.add_design_var('f_struct', indices=[0, 12, 34, 40], lower=-5, upper=10, ref=10.0)
        #prob.model.add_design_var('mesh.x_struct0', indices=[0, 2, 5, 10], lower=-5, upper=10, ref=10.0)

        # objectives and nonlinear constraints
        #prob.model.add_objective('analysis.mass', ref=100.0)
        #prob.model.add_constraint('analysis.func_struct', ref=1.0, upper=1.0)

        prob.setup(mode='rev', force_alloc_complex=True)
        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        print('----------------starting check totals--------------')
        data = self.prob.check_totals(of=['analysis.func_struct','analysis.mass'],
                                      wrt='mesh.x_struct0', method='cs',
                                      step=1e-30, step_calc='rel')
        #data = self.prob.check_totals(wrt='mesh.x_struct0', method='cs',
        #                              step=1e-30, step_calc='rel')
        for var, err in data.items():
            rel_err = err['rel error']
            assert_near_equal(rel_err.forward, 0.0, 1e-8)

        data = self.prob.check_totals(of=['analysis.func_struct'], wrt='f_struct',
                                      method='cs', step=1e-30, step_calc='rel')
        for var, err in data.items():
            rel_err = err['rel error']
            assert_near_equal(rel_err.forward, 0.0, 2e-8)

        data = self.prob.check_totals(of=['analysis.func_struct','analysis.mass'],
                                      wrt='dv_struct', method='cs',
                                      step=1e-30, step_calc='rel')
        #data = self.prob.check_totals(wrt='dv_struct', method='cs',
        #                              step=1e-30, step_calc='rel')
        for var, err in data.items():
            rel_err = err['rel error']
            assert_near_equal(rel_err.forward, 0.0, 5e-8)


if __name__ == '__main__':
    unittest.main()
