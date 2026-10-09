import json
import unittest

import numpy as np
import openmdao.api as om
from fake_remote import (
    PARALLEL_SCENARIO_ADDITIONAL_INPUTS,
    PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS,
    PARALLEL_SCENARIO_RESPONSES,
    InProcessServer,
    get_parallel_scenarios_group,
)
from mpi4py import MPI
from openmdao.utils.assert_utils import assert_near_equal

DESIGN_VARS = ["x", "y"]


def initialize_message():
    return "initialize|" + json.dumps(
        {
            "additional_inputs": PARALLEL_SCENARIO_ADDITIONAL_INPUTS,
            "additional_outputs": PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS,
            "additional_constants": [],
            "component_name": "remote",
        }
    )


def design_message(command, x, y, p):
    return f"{command}|" + json.dumps(
        {
            "design_vars": {"x": {"val": list(x)}, "y": {"val": [y]}},
            "additional_inputs": {"p": {"val": [p]}},
            "additional_constants": {},
            "additional_outputs": PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS,
            "component_name": "remote",
        }
    )


def response_type(name):
    return "objective" if name == "m" else "constraints"


class ServerTestMixin:
    """
    Compares a Server running the parallel-scenario model against the same
    model evaluated directly with OpenMDAO on the same communicator.
    """

    comm = MPI.COMM_SELF

    def setUp(self):
        self.server = InProcessServer(get_parallel_scenarios_group, comm=self.comm)
        self.reference = om.Problem(get_parallel_scenarios_group(), comm=self.comm)
        self.reference.setup(mode="rev")

    def _handle(self, message):
        # collective over the server comm; the reply only exists on rank 0
        reply = self.server.handle(message if self.comm.rank == 0 else None)
        return self.comm.bcast(json.loads(reply) if self.comm.rank == 0 else None)

    def _reference_values(self, x, y, p):
        self.reference.set_val("x", np.array(x))
        self.reference.set_val("y", y)
        self.reference.set_val("p", p)
        self.reference.run_model()
        names = PARALLEL_SCENARIO_RESPONSES + PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS
        return {name: self.reference.get_val(name, get_remote=True) for name in names}

    def _reference_totals(self):
        return self.reference.compute_totals(
            of=PARALLEL_SCENARIO_RESPONSES + PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS,
            wrt=DESIGN_VARS + PARALLEL_SCENARIO_ADDITIONAL_INPUTS,
        )

    def _assert_values_match(self, reply, expected):
        for name in PARALLEL_SCENARIO_RESPONSES:
            assert_near_equal(
                reply[response_type(name)][name]["val"], expected[name], tolerance=1e-12
            )
        for name in PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS:
            assert_near_equal(
                reply["additional_outputs"][name]["val"],
                expected[name],
                tolerance=1e-12,
            )

    def test_initialize_reports_all_scenarios(self):
        reply = self._handle(initialize_message())
        self.assertEqual(sorted(reply["design_vars"]), DESIGN_VARS)
        self.assertEqual(list(reply["objective"]), ["m"])
        self.assertEqual(
            sorted(reply["constraints"]),
            ["par.scen0.c", "par.scen0.s", "par.scen1.c", "par.scen1.s"],
        )
        self.assertEqual(
            sorted(reply["additional_outputs"]), PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS
        )
        self.assertEqual(
            sorted(reply["additional_inputs"]), PARALLEL_SCENARIO_ADDITIONAL_INPUTS
        )
        self._assert_values_match(reply, self._reference_values([1.0, 2.0], 0.5, 1.5))

    def test_evaluate_new_design(self):
        self._handle(initialize_message())
        reply = self._handle(design_message("evaluate", [0.3, -1.2], 2.0, -0.7))
        self._assert_values_match(reply, self._reference_values([0.3, -1.2], 2.0, -0.7))

    def test_design_change_detection(self):
        self._handle(initialize_message())
        counter = self.server.design_counter
        self._handle(design_message("evaluate", [0.3, -1.2], 2.0, -0.7))
        self.assertEqual(self.server.design_counter, counter + 1)
        self._handle(design_message("evaluate", [0.3, -1.2], 2.0, -0.7))  # same design
        self.assertEqual(self.server.design_counter, counter + 1)
        self._handle(
            design_message("evaluate", [0.3, -1.2], 2.0, 4.0)
        )  # only p changes
        self.assertEqual(self.server.design_counter, counter + 2)
        # every rank must agree on whether to re-run the (collective) model
        self.assertEqual(len(set(self.comm.allgather(self.server.design_counter))), 1)

    def test_derivatives_with_parallel_deriv_coloring(self):
        self._handle(initialize_message())
        reply = self._handle(
            design_message("evaluate derivatives", [0.3, -1.2], 2.0, -0.7)
        )
        self._assert_values_match(reply, self._reference_values([0.3, -1.2], 2.0, -0.7))
        totals = self._reference_totals()
        for name in PARALLEL_SCENARIO_RESPONSES:
            derivs = reply[response_type(name)][name]["derivatives"]
            for wrt in DESIGN_VARS + PARALLEL_SCENARIO_ADDITIONAL_INPUTS:
                assert_near_equal(derivs[wrt], totals[name, wrt], tolerance=1e-10)
        for name in PARALLEL_SCENARIO_ADDITIONAL_OUTPUTS:
            derivs = reply["additional_outputs"][name]["derivatives"]
            for wrt in DESIGN_VARS + PARALLEL_SCENARIO_ADDITIONAL_INPUTS:
                assert_near_equal(derivs[wrt], totals[name, wrt], tolerance=1e-10)
        # coloring was recomputed for the server's of/wrt (incl. additional variables)
        self.assertIsNotNone(self.server.coloring_info)

    def test_repeated_derivatives_reuse_coloring_and_results(self):
        self._handle(initialize_message())
        message = design_message("evaluate derivatives", [0.3, -1.2], 2.0, -0.7)
        first = self._handle(message)
        coloring = self.server.coloring_info
        second = self._handle(message)
        self.assertIs(self.server.coloring_info, coloring)
        self.assertEqual(first["constraints"], second["constraints"])
        third = self._handle(
            design_message("evaluate derivatives", [1.0, 1.0], -1.0, 0.2)
        )
        self._reference_values([1.0, 1.0], -1.0, 0.2)
        totals = self._reference_totals()
        assert_near_equal(
            third["constraints"]["par.scen1.c"]["derivatives"]["y"],
            totals["par.scen1.c", "y"],
            tolerance=1e-10,
        )


class TestServerSerial(ServerTestMixin, unittest.TestCase):
    comm = MPI.COMM_SELF


@unittest.skipUnless(
    MPI.COMM_WORLD.size == 2,
    "requires 2 MPI processes (run with testflo, or mpiexec -n 2 python -m unittest)",
)
class TestServerParallel(ServerTestMixin, unittest.TestCase):
    N_PROCS = 2
    comm = MPI.COMM_WORLD

    def test_scenarios_are_on_different_ranks(self):
        # the point of the parallel test: each scenario only exists on one rank,
        # so the server must gather the other rank's values
        # _get_subsystem also finds subsystems owned by other ranks
        local = [s.name for s in self.server.prob.model.par._subsystems_myproc]
        self.assertEqual(len(local), 1)
        self.assertEqual(
            sorted(sum(self.comm.allgather(local), [])), ["scen0", "scen1"]
        )


if __name__ == "__main__":
    unittest.main()
