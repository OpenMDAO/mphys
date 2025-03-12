import unittest
from distutils.version import LooseVersion

import numpy as np
import openmdao
import openmdao.api as om
from common_methods import CommonMethods
from mpi4py import MPI
from openmdao.utils.assert_utils import assert_near_equal

from mphys import DistributedSummer, DistributedVariableDescription


class TestDistributedSummer(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        inputs = self.prob.model.add_subsystem(
            "inputs", om.IndepVarComp(), promotes=["*"]
        )

        inputs.add_output("dist_input1", val=np.ones(10, dtype=float), distributed=True)
        inputs.add_output(
            "dist_input2", val=np.arange(10, dtype=float), distributed=True
        )

        # Create a mask that masks every other entry of the input array
        input_metadata = [
            DistributedVariableDescription(
                "dist_input1", shape=10, tags=["mphys_coordinates"]
            ),
            DistributedVariableDescription(
                "dist_input2", shape=10, tags=["mphys_coordinates"]
            ),
        ]
        output_metadata = DistributedVariableDescription(
            "sumed_output", shape=10, tags=["mphys_coordinates"]
        )
        sumer = DistributedSummer(inputs=input_metadata, output=output_metadata)

        self.prob.model.add_subsystem("sumer", sumer, promotes=["*"])

        self.prob.setup(force_alloc_complex=True)

    def test_run_model(self):
        self.common.test_run_model(self, write_n2=False)

    def test_check_partials(self):
        partials = self.prob.check_partials(compact_print=True, method="cs")
        tol = 1e-9

        check_error = partials["sumer"][("sumed_output", "dist_input1")]["abs error"]
        assert_near_equal(check_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(check_error.forward, 0.0, tolerance=tol)
        assert_near_equal(check_error.fwd_rev, 0.0, tolerance=tol)

        check_error = partials["sumer"][("sumed_output", "dist_input2")]["abs error"]
        assert_near_equal(check_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(check_error.forward, 0.0, tolerance=tol)
        assert_near_equal(check_error.fwd_rev, 0.0, tolerance=tol)


if __name__ == "__main__":
    unittest.main()
