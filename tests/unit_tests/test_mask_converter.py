import unittest

import numpy as np
import openmdao.api as om
from common_methods import CommonMethods
from mpi4py import MPI
from openmdao.utils.assert_utils import assert_near_equal

from mphys import MaskedConverter, MaskedVariableDescription, UnmaskedConverter


class TestMaskConverterSingle(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        inputs = self.prob.model.add_subsystem("inputs", om.IndepVarComp())

        inputs.add_output(
            "unmasked_input", val=np.ones(10, dtype=float), distributed=True
        )
        inputs.add_output(
            "masked_input", val=np.arange(5, dtype=float), distributed=True
        )

        # Create a mask that masks every other entry of the input array
        mask_input = MaskedVariableDescription(
            "unmasked_input", shape=10, tags=["mphys_coordinates"]
        )
        mask_output = MaskedVariableDescription(
            "masked_output", shape=5, tags=["mphys_coordinates"]
        )
        mask = np.zeros([10], dtype=bool)
        mask[:] = True
        mask[::2] = False
        masker = MaskedConverter(
            input=mask_input, output=mask_output, mask=mask, distributed=True
        )

        self.prob.model.add_subsystem("masker", masker)

        unmask_input = MaskedVariableDescription(
            "masked_input", shape=5, tags=["mphys_coordinates"]
        )
        unmask_output = MaskedVariableDescription(
            "unmasked_output", shape=10, tags=["mphys_coordinates"]
        )
        unmasker = UnmaskedConverter(
            input=unmask_input,
            output=unmask_output,
            mask=mask,
            distributed=True,
            default_values=1.0,
        )

        self.prob.model.add_subsystem("unmasker", unmasker)

        self.prob.model.connect("inputs.unmasked_input", "masker.unmasked_input")
        self.prob.model.connect("inputs.masked_input", "unmasker.masked_input")

        self.prob.setup(force_alloc_complex=True)

    def test_run_model(self):
        self.common.test_run_model(self, write_n2=False)

    def test_check_partials(self):
        partials = self.prob.check_partials(compact_print=True, method="cs")
        tol = 1e-9

        rel_error = partials["masker"][("masked_output", "unmasked_input")]["rel error"]
        assert_near_equal(rel_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward_reverse, 0.0, tolerance=tol)

        rel_error = partials["unmasker"][("unmasked_output", "masked_input")][
            "rel error"
        ]
        assert_near_equal(rel_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward_reverse, 0.0, tolerance=tol)


class TestMaskConverterMulti(unittest.TestCase):
    N_PROCS = 1  # TODO should be 2 or more but there is a bug in OM currently

    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        inputs = self.prob.model.add_subsystem("inputs", om.IndepVarComp())

        inputs.add_output(
            "unmasked_input", val=np.ones(10, dtype=float), distributed=True
        )
        inputs.add_output(
            "masked_input_1", val=np.arange(5, dtype=float), distributed=True
        )
        inputs.add_output(
            "masked_input_2", val=np.arange(5, dtype=float), distributed=True
        )

        # Create a mask that masks every other entry of the input array
        mask_input = MaskedVariableDescription(
            "unmasked_input", shape=10, tags=["mphys_coordinates"]
        )
        mask_output = [
            MaskedVariableDescription(
                "masked_output_1", shape=5, tags=["mphys_coordinates"]
            ),
            MaskedVariableDescription(
                "masked_output_2", shape=5, tags=["mphys_coordinates"]
            ),
        ]
        mask = [
            np.zeros([10], dtype=bool),
            np.zeros([10], dtype=bool),
        ]
        mask[0][0:5] = True
        mask[0][5:10] = False
        mask[1][0:5] = False
        mask[1][5:10] = True
        masker = MaskedConverter(
            input=mask_input, output=mask_output, mask=mask, distributed=True
        )

        self.prob.model.add_subsystem("masker", masker)

        unmask_input = [
            MaskedVariableDescription(
                "masked_input_1", shape=5, tags=["mphys_coordinates"]
            ),
            MaskedVariableDescription(
                "masked_input_2", shape=5, tags=["mphys_coordinates"]
            ),
        ]
        unmask_output = MaskedVariableDescription(
            "unmasked_output", shape=10, tags=["mphys_coordinates"]
        )
        unmasker = UnmaskedConverter(
            input=unmask_input,
            output=unmask_output,
            mask=mask,
            distributed=True,
            default_values=1.0,
        )

        self.prob.model.add_subsystem("unmasker", unmasker)

        self.prob.model.connect("inputs.unmasked_input", "masker.unmasked_input")
        self.prob.model.connect("inputs.masked_input_1", "unmasker.masked_input_1")
        self.prob.model.connect("inputs.masked_input_2", "unmasker.masked_input_2")

        self.prob.setup(force_alloc_complex=True)

    def test_run_model(self):
        self.common.test_run_model(self, write_n2=False)

    def test_check_partials(self):
        partials = self.prob.check_partials(compact_print=True, method="cs")
        tol = 1e-9

        rel_error = partials["masker"][("masked_output_1", "unmasked_input")][
            "rel error"
        ]
        assert_near_equal(rel_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward, 0.0, tolerance=tol)
        assert_near_equal(rel_error.fwd_rev, 0.0, tolerance=tol)
        rel_error = partials["masker"][("masked_output_2", "unmasked_input")][
            "rel error"
        ]
        assert_near_equal(rel_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward, 0.0, tolerance=tol)
        assert_near_equal(rel_error.fwd_rev, 0.0, tolerance=tol)

        rel_error = partials["unmasker"][("unmasked_output", "masked_input_1")][
            "rel error"
        ]
        assert_near_equal(rel_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward, 0.0, tolerance=tol)
        assert_near_equal(rel_error.fwd_rev, 0.0, tolerance=tol)
        rel_error = partials["unmasker"][("unmasked_output", "masked_input_2")][
            "rel error"
        ]
        assert_near_equal(rel_error.reverse, 0.0, tolerance=tol)
        assert_near_equal(rel_error.forward, 0.0, tolerance=tol)
        assert_near_equal(rel_error.fwd_rev, 0.0, tolerance=tol)


if __name__ == "__main__":
    unittest.main()
