import unittest

import openmdao.api as om
from common_methods import CommonMethods

from mphys.core import MPhysGroup


class Comp(om.ExplicitComponent):
    def setup(self):
        self.add_input("dv_in", tags=["mphys_input"])
        self.add_output("dv_out", tags=["mphys_input"])

        self.add_input("result_in", tags=["mphys_result"])
        self.add_output("result_out", tags=["mphys_result"])

        self.add_input("coupling_in", tags=["mphys_coupling"])
        self.add_output("coupling_out", tags=["mphys_coupling"])

        self.add_input("coordinates_in", tags=["mphys_coordinates"])
        self.add_output("coordinates_out", tags=["mphys_coordinates"])


class TestMPhysGroupAddingMphysSubsystem(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        group = self.prob.model.add_subsystem("mphys_group", MPhysGroup())
        group.mphys_add_subsystem("comp1", Comp())

        self.prob.setup()

    def test_run_model(self):
        self.common.test_run_model(self, write_n2=False)

    def test_tagged_input_promotion(self):
        inputs = self.prob.model.mphys_group.get_io_metadata(
            "input", metadata_keys=["tags"]
        )
        for key, val in inputs.items():
            if (
                "mphys_input" in val["tags"]
                or "mphys_coupling" in val["tags"]
                or "mphys_coordinates" in val["tags"]
            ):
                self.assertEqual(val["prom_name"], key.split(".")[-1])
            else:
                self.assertEqual(val["prom_name"], key)

    def test_tagged_output_promotion(self):
        inputs = self.prob.model.mphys_group.get_io_metadata(
            "output", metadata_keys=["tags"]
        )
        for key, val in inputs.items():
            if (
                "mphys_result" in val["tags"]
                or "mphys_coupling" in val["tags"]
                or "mphys_coordinates" in val["tags"]
            ):
                self.assertEqual(val["prom_name"], key.split(".")[-1])
            else:
                self.assertEqual(val["prom_name"], key)


class TestMPhysGroupNotAddingMphysSubsystem(unittest.TestCase):
    """
    Subsystem with inputs/outputs with mphys tags but not added with mphys_add_subsystem
    should not promote tagged variables
    """

    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        group = self.prob.model.add_subsystem("mphys_group", MPhysGroup())
        group.add_subsystem("comp1", Comp())

        self.prob.setup()

    def test_run_model(self):
        self.common.test_run_model(self, write_n2=False)

    def test_input_not_promoted(self):
        inputs = self.prob.model.mphys_group.get_io_metadata(
            "input", metadata_keys=["tags"]
        )
        for key, val in inputs.items():
            self.assertEqual(val["prom_name"], key)

    def test_output_not_promoted(self):
        inputs = self.prob.model.mphys_group.get_io_metadata(
            "output", metadata_keys=["tags"]
        )
        for key, val in inputs.items():
            self.assertEqual(val["prom_name"], key)


if __name__ == "__main__":
    unittest.main()
