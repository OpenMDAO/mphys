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


class Top(om.Group):
    def setup(self):

        ################################################################################
        # STRUCT
        ################################################################################

        # creating the options dictionary is same for both solvers
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

        def f5_writer(tacs):
            flag = TACS.ToFH5.NODES | TACS.ToFH5.DISPLACEMENTS | TACS.ToFH5.STRAINS | TACS.ToFH5.EXTRAS
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile(os.path.join(baseDir, '../output_files/wingbox.f5'))

        # common setup options
        struct_options = {
            'add_elements': add_elements,
            'get_funcs': get_funcs,
            'mesh_file': os.path.join(baseDir, '../input_files/debug.bdf'),
            # 'f5_writer'   : f5_writer,
        }

        struct_builder = TacsBuilder(struct_options)


        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc for DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # create the multiphysics multipoint group.
        mp = self.add_subsystem('mp', Multipoint(aero_builder=None, struct_builder=struct_builder, xfer_builder=None))

        mp.mphys_add_scenario('s0')

    def configure(self):

        # structural DVs are shared across TACS and modal solver
        ndv_struct = self.mp.struct_builder.get_ndv()

        # a flat DV array w/o any mapping
        self.dvs.add_output('dv_struct', np.array(ndv_struct * [0.01]))

        self.connect('dv_struct', ['mp.s0.solver_group.struct.dv_struct', 'mp.s0.struct_funcs.dv_struct'])

        f_size = self.mp.s0.solver_group.struct.solver.ans.getArray().size
        self.dvs.add_output('f_struct', np.ones(f_size))

        self.connect('f_struct', ['mp.s0.solver_group.struct.f_struct'])

        self.mp.struct_mesh.mphys_add_coordinate_input()
        self.dvs.add_output('xpts', self.mp.struct_mesh.xpts.getArray())

        self.connect('xpts', ['mp.struct_mesh.x_struct0_points'])


class TestTACs(unittest.TestCase):
    def setUp(self):

        ################################################################################
        # OpenMDAO setup
        ################################################################################
        prob = om.Problem()
        prob.model = Top()

        # DVs

        # objectives and nonlinear constraints
        prob.model.add_objective('mp.s0.struct_funcs.mass', ref=100.0)
        prob.model.add_constraint('mp.s0.struct_funcs.funcs.func_struct', ref=1.0, upper=1.0)

        prob.model.add_design_var('dv_struct', indices=[0], lower=-5, upper=10, ref=10.0)
        prob.model.add_design_var('f_struct', indices=[0, 12, 34, 40], lower=-5, upper=10, ref=10.0)
        prob.model.add_design_var('xpts', indices=[0, 2, 5, 10], lower=-5, upper=10, ref=10.0)

        prob.setup(mode='rev',force_alloc_complex=True)
        # om.n2(
        #     prob,
        #     show_browser=False,
        #     outfile='test_struct_derivs.html' ,
        # )

        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        print('----------------strating check totals--------------')
        data = self.prob.check_totals(wrt='xpts', method='cs', step=1e-30, step_calc='rel')  # out_stream=None

        for var, err in data.items():

            rel_err = err['rel error']  # ,  'rel error']
            assert_near_equal(rel_err.forward, 0.0, 1e-8)
        data = self.prob.check_totals(of=['mp.s0.struct_funcs.funcs.func_struct'], wrt='f_struct', method='cs', step=1e-30, step_calc='rel')  # out_stream=None
        for var, err in data.items():

            rel_err = err['rel error']  # ,  'rel error']
            assert_near_equal(rel_err.forward, 0.0, 1e-8)
        data = self.prob.check_totals(wrt='dv_struct', method='cs', step=1e-30, step_calc='rel')  # out_stream=None
        for var, err in data.items():

            rel_err = err['rel error']  # ,  'rel error']
            assert_near_equal(rel_err.forward, 0.0, 5e-8)


if __name__ == '__main__':
    unittest.main()
