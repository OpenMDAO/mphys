# --- Python 3.8 ---
"""
@File    :   test_meld.py
@Time    :   2020/12/20
@Author  :   Josh Anibal
@Desc    :   tests the output and derivatives of meld. Based on Kevin's test script
"""

# === Standard Python modules ===
import unittest
import os

# === External Python modules ===
import numpy as np
from mpi4py import MPI

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys import Builder
from pyfuntofem import MeldBuilder


isym = 1

class DummyBuilder(Builder):
    def __init__(self, num_nodes, ndof):
        self.num_nodes = num_nodes
        self.ndof = ndof
    def get_number_of_nodes(self):
        return self.num_nodes
    def get_ndof(self):
        return self.ndof

class TestXferClasses(unittest.TestCase):
    def setUp(self):
        class FakeStructMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('x_struct0', shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs['x_struct0'] = self.nodes

        class FakeStructDisps(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)
                self.nodes = np.arange(12)

            def setup(self):
                self.add_output('u_struct', shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs['u_struct'] = self.nodes

        class FakeAeroLoads(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('f_aero', shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs['f_aero'] = self.nodes

        class FakeAeroMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output('x_aero', shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs['x_aero'] = self.nodes

        np.random.seed(0)
        aero_builder = DummyBuilder(4,3)
        struct_builder = DummyBuilder(4,3)
        meld_builder = MeldBuilder(aero_builder, struct_builder, isym=1, check_partials=True)
        meld_builder.initialize(MPI.COMM_WORLD)

        prob = om.Problem()
        prob.model.add_subsystem('aero_mesh', FakeAeroMesh())
        prob.model.add_subsystem('struct_mesh', FakeStructMesh())
        prob.model.add_subsystem('struct_disps', FakeStructDisps())
        prob.model.add_subsystem('aero_loads', FakeAeroLoads())

        disp, load = meld_builder.get_coupling_group_subsystem()
        prob.model.add_subsystem('disp_xfer',disp)
        prob.model.add_subsystem('load_xfer',load)

        prob.model.connect('aero_mesh.x_aero', ['disp_xfer.x_aero0', 'load_xfer.x_aero0'])
        prob.model.connect('struct_mesh.x_struct0', ['disp_xfer.x_struct0', 'load_xfer.x_struct0'])
        prob.model.connect('struct_disps.u_struct', ['disp_xfer.u_struct', 'load_xfer.u_struct'])
        prob.model.connect('aero_loads.f_aero', ['load_xfer.f_aero'])

        prob.model.add_subsystem('aero_mesh2',FakeAeroMesh())
        prob.model.add_subsystem('struct_mesh2',FakeStructMesh())
        prob.model.add_subsystem('struct_disps2',FakeStructDisps())
        prob.model.add_subsystem('aero_loads2',FakeAeroLoads())

        disp, load = meld_builder.get_coupling_group_subsystem()
        prob.model.add_subsystem('disp_xfer2',disp)
        prob.model.add_subsystem('load_xfer2',load)

        prob.model.connect('aero_mesh2.x_aero',['disp_xfer2.x_aero0','load_xfer2.x_aero0'])
        prob.model.connect('struct_mesh2.x_struct0',['disp_xfer2.x_struct0','load_xfer2.x_struct0'])
        prob.model.connect('struct_disps2.u_struct',['disp_xfer2.u_struct','load_xfer2.u_struct'])
        prob.model.connect('aero_loads2.f_aero',['load_xfer2.f_aero'])

        prob.setup(force_alloc_complex=True)
        self.prob = prob
        #om.n2(prob, show_browser=False, outfile='test.html')

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        data = self.prob.check_partials(compact_print=True, method='cs')

        # there is an openmdao util to check partiales, but we can't use it
        # because only SOME of the fwd derivatives are implemented
        for key, comp in data.items():
            for var, err in comp.items():
                rel_err = err['rel error']
                assert_near_equal(rel_err.reverse, 0.0, 1e-12)
                if var[1] == 'f_aero' or var[1] == 'u_struct':
                    assert_near_equal(rel_err.forward, 0.0, 1e-12)


if __name__ == '__main__':
    unittest.main()
