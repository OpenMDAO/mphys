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
from parameterized import parameterized, parameterized_class

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from baseclasses import AeroProblem

from mphys.mphys_meld import MeldBuilder
from mphys.mphys_rlt import RltBuilder

import mphys


meld_options = {"isym": 1, "n": 200, "beta": 0.5}
rlt_options = {"transfergaussorder": 2}


@parameterized_class(
    ("name", "xfer_builder_class", "xfer_options"),
    [
        ("meld", MeldBuilder, meld_options),
        #    ('rlt', RltBuilder, rlt_options) # RLT can't take a dummy solver as input. It will be tested in regression tests
    ],
)
class TestXferClasses(unittest.TestCase):
    def setUp(self):
        class FakeStructMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output("x_s", shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs["x_s"] = self.nodes

        class FakeStructDisps(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)
                self.nodes = np.arange(12)

            def setup(self):
                self.add_output("u_s", shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs["u_s"] = self.nodes

        class FakeAeroLoads(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output("f_a", shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs["f_a"] = self.nodes

        class FakeAeroMesh(om.ExplicitComponent):
            def initialize(self):
                self.nodes = np.random.rand(12)

            def setup(self):
                self.add_output("x_a", shape=self.nodes.size)

            def compute(self, inputs, outputs):
                outputs["x_a"] = self.nodes

        np.random.seed(0)
        prob = om.Problem()
        aero_mesh = prob.model.add_subsystem("aero_mesh", FakeAeroMesh())
        prob.model.add_subsystem("struct_mesh", FakeStructMesh())
        prob.model.add_subsystem("struct_disps", FakeStructDisps())
        prob.model.add_subsystem("aero_loads", FakeAeroLoads())

        # if self.name == 'rlt':
        #     class DummySolver(object):
        #         pass

        #     aero_solver = DummySolver()
        #     def getSurfaceCoordinates(self, groupName=None, includeZipper=True):
        #         return aero_mesh.nodes

        #     def getSurfaceConnectivity(self, groupName=None, includeZipper=True, includeCGNS=False):
        #         ncell = 8
        #         conn =  np.random.randint(0, 14, size=(ncell, 4))
        #         faceSizes = 4*np.ones(len(conn), 'intc')

        #         return conn, faceSizes

        #     aero_solver.getSurfaceCoordinates = getSurfaceCoordinates
        #     aero_solver.getSurfaceConnectivity = getSurfaceConnectivity
        #     aero_solver.allWallsGroup = None
        #     aero_solver.comm = MPI.COMM_WORLD

        #     aero_builder = mphys.DummyBuilder(nnodes=4, solver=aero_solver)

        #     struct_builder = mphys.DummyBuilder(nnodes=4, ndof=3, solver=None)
        # else:
        aero_builder = mphys.DummyBuilder(nnodes=4)

        struct_builder = mphys.DummyBuilder(nnodes=4, ndof=3)

        self.builder = self.xfer_builder_class(self.xfer_options, aero_builder, struct_builder, check_partials=True)

        self.builder.init_xfer_object(MPI.COMM_WORLD)
        disp_xfer, load_xfer = self.builder.get_element()

        disp = prob.model.add_subsystem("disp_xfer", disp_xfer)
        load = prob.model.add_subsystem("load_xfer", load_xfer)

        prob.model.connect("aero_mesh.x_a", ["disp_xfer.x_a0", "load_xfer.x_a0"])
        prob.model.connect("struct_mesh.x_s", ["disp_xfer.x_s0", "load_xfer.x_s0"])
        prob.model.connect("struct_disps.u_s", ["disp_xfer.u_s", "load_xfer.u_s"])
        prob.model.connect("aero_loads.f_a", ["load_xfer.f_a"])

        prob.setup(force_alloc_complex=True)
        self.prob = prob
        # om.n2(prob, show_browser=False, outfile='test.html')

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        data = self.prob.check_partials(out_stream=None, compact_print=False)

        # there is an openmdao util to check partiacles, but we can't use it
        # because only SOME of the fwd derivatives are implemented
        for key, comp in data.items():
            for var, err in comp.items():

                rel_err = err["rel error"]  # ,  'rel error']

                assert_near_equal(rel_err.reverse, 0.0, 5e-6)

                if var[1] == "f_a" or var[1] == "u_s":
                    assert_near_equal(rel_err.forward, 0.0, 5e-6)


if __name__ == "__main__":
    unittest.main()