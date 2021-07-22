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
from mphys.mphys_meld_lfd import MeldLfdBuilder


isym = 1
num_nodes = 4
num_modes = 3

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
        class FakeStructMesh(om.IndepVarComp):
            def setup(self):
                self.add_output('x_struct0', val=np.array(np.random.rand(num_nodes*3),order='C'),
                                distributed=True)

        class FakeStructModes(om.IndepVarComp):
            def setup(self):
                self.add_output('mode_shapes_struct', val=np.array(np.random.rand(num_nodes*3,num_modes),order='C'),
                                 distributed=True)

        class FakeAeroMesh(om.IndepVarComp):
            def setup(self):
                val = np.array(np.random.rand(num_nodes*3),order='C')
                self.add_output('x_aero', val= val, distributed=True)

        np.random.seed(0)
        aero_builder = DummyBuilder(4,3)
        struct_builder = DummyBuilder(4,3)
        meld_builder = MeldLfdBuilder(aero_builder, struct_builder, num_modes, isym=1, check_partials=True)
        meld_builder.initialize(MPI.COMM_WORLD)

        prob = om.Problem()
        prob.model.add_subsystem('aero_mesh', FakeAeroMesh())
        prob.model.add_subsystem('struct_mesh', FakeStructMesh())
        prob.model.add_subsystem('modal', FakeStructModes())

        mode_xfer = meld_builder.get_post_coupling_subsystem()
        prob.model.add_subsystem('mode_xfer',mode_xfer)

        prob.model.connect('aero_mesh.x_aero', 'mode_xfer.x_aero0')
        prob.model.connect('struct_mesh.x_struct0', 'mode_xfer.x_struct0')
        prob.model.connect('modal.mode_shapes_struct', 'mode_xfer.mode_shapes_struct')

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.final_setup()
        #prob.set_complex_step_mode(True)
        self.prob = prob
        om.n2(prob, show_browser=False, outfile='test.html')

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        data = self.prob.check_partials(compact_print=True, method='cs')

        # there is an openmdao util to check partials, but we can't use it
        # because only SOME of the fwd derivatives are implemented
        for key, comp in data.items():
            for var, err in comp.items():
                rel_err = err['rel error']
                assert_near_equal(rel_err.reverse, 0.0, 1e-12)


if __name__ == '__main__':
    unittest.main()
