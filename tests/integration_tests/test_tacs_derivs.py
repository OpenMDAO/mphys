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

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys.multipoint import Multipoint
from mphys.scenario_structural import ScenarioStructural

from tacs import elements, constitutive, functions
from mphys.mphys_tacs import TacsBuilder


baseDir = os.path.dirname(os.path.abspath(__file__))

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    rho = 2780.0  # density, kg/m^3
    E = 73.1e9  # elastic modulus, Pa
    nu = 0.33  # poisson's ratio
    ys = 324.0e6  # yield stress, Pa
    thickness = 0.012
    min_thickness = 0.002
    max_thickness = 0.05

    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=thickness, tNum=dvNum, tlb=min_thickness, tub=max_thickness)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None
    elem = elements.Quad4Shell(transform, con)

    return elem

class Top(Multipoint):

    def setup(self):

        tacs_options = {'element_callback' : element_callback,
                        'mesh_file': '../input_files/debug.bdf'}

        tacs_builder = TacsBuilder(tacs_options, check_partials=True, coupled=True, write_solution=False)
        tacs_builder.initialize(self.comm)
        ndv_struct = tacs_builder.get_ndv()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))

        f_size = tacs_builder.get_ndof() * tacs_builder.get_number_of_nodes()
        forces = self.add_subsystem('forces', om.IndepVarComp(), promotes=['*'])
        forces.add_output('f_struct', val=np.ones(f_size), distributed=True)

        self.add_subsystem('mesh', tacs_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=tacs_builder))
        self.connect('mesh.x_struct0', 'analysis.x_struct0')
        self.connect('dv_struct', 'analysis.dv_struct')
        self.connect('f_struct', 'analysis.f_struct')

    def configure(self):
        # create the tacs problems for adding evalfuncs and fixed structural loads to the analysis point.
        # This is custom to the tacs based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the struct group.
        fea_assembler = self.analysis.coupling.fea_assembler

        # ==============================================================================
        # Setup structural problem
        # ==============================================================================
        # Structural problem
        # Set converges to be tight for test
        prob_options = {'L2Convergence': 1e-20, 'L2ConvergenceRel': 1e-20}
        sp = fea_assembler.createStaticProblem(name='test', options=prob_options)
        # Add TACS Functions
        sp.addFunction('mass', functions.StructuralMass)
        sp.addFunction('ks_vmfailure', functions.KSFailure, ksWeight=50.0)

        self.analysis.coupling.mphys_set_sp(sp)
        self.analysis.struct_post.mphys_set_sp(sp)


class TestTACS(unittest.TestCase):
    N_PROCS=2
    def setUp(self):
        prob = om.Problem()
        prob.model = Top()

        prob.setup(mode='rev', force_alloc_complex=True)
        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        print('----------------starting check totals--------------')
        data = self.prob.check_totals(of=['analysis.struct_post.ks_vmfailure', 'analysis.struct_post.mass'],
                                      wrt='mesh.fea_mesh.x_struct0', method='cs',
                                      step=1e-50, step_calc='rel')
        for var, err in data.items():
            rel_err = err['rel error']
            assert_near_equal(rel_err.forward, 0.0, 2e-8)

        data = self.prob.check_totals(of=['analysis.struct_post.ks_vmfailure'], wrt='f_struct',
                                      method='cs', step=1e-50, step_calc='rel')
        for var, err in data.items():
            rel_err = err['rel error']
            assert_near_equal(rel_err.forward, 0.0, 5e-8)

        data = self.prob.check_totals(of=['analysis.struct_post.ks_vmfailure', 'analysis.struct_post.mass'],
                                      wrt='dv_struct', method='cs',
                                      step=1e-50, step_calc='rel')
        for var, err in data.items():
            rel_err = err['rel error']
            assert_near_equal(rel_err.forward, 0.0, 1e-8)


if __name__ == '__main__':
    unittest.main()
