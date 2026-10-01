import json
import os
import tempfile
import unittest

import numpy as np
import openmdao.api as om
from fake_remote import (
    InProcessRemoteComp,
    InProcessServer,
    RecordingServerManager,
    UnreachableServer,
    get_nested_group,
    get_paraboloid_group,
)
from mpi4py import MPI
from openmdao.utils.assert_utils import assert_near_equal

from mphys.network import Server


def expected_outputs(x, y, p, c):
    x = np.asarray(x, dtype=float)
    return {
        "f": x[0] ** 2 + x[1] ** 2 + y**2 + p,
        "g": x[0] + x[1] + c,
        "h": y - x[0],
        "z": 3 * x + y,
    }


def build_problem(server_factory=None, server_manager=None, **remote_options):
    if server_factory is None:
        server_factory = lambda: InProcessServer(get_paraboloid_group)  # noqa: E731
    prob = om.Problem()
    prob.model.add_subsystem(
        "remote",
        InProcessRemoteComp(
            server_factory=server_factory,
            server_manager=server_manager,
            additional_remote_inputs=["p"],
            additional_remote_outputs=["z"],
            additional_remote_constants=["c"],
            **remote_options,
        ),
        promotes=["*"],
    )
    return prob


class TestRemoteCompSetup(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        self.prob = build_problem()
        self.prob.setup()
        self.prob.final_setup()
        self.remote = self.prob.model.remote
        self.server_model = self.remote.server.prob.model

    def _assert_meta_matches_server(self, client_meta, server_meta, keys):
        defaults = {"ref": 1.0, "ref0": 0.0, "scaler": 1.0, "adder": 0.0}
        for key in keys:
            client_val = client_meta[key]
            server_val = server_meta[key]
            if key in defaults:
                client_val = defaults[key] if client_val is None else client_val
                server_val = defaults[key] if server_val is None else server_val
            if server_val is None:
                self.assertIsNone(client_val, key)
            else:
                assert_near_equal(client_val, server_val, tolerance=1e-12)

    def test_inputs_and_outputs_created(self):
        inputs = self.remote.list_inputs(out_stream=None, prom_name=True)
        outputs = self.remote.list_outputs(out_stream=None, prom_name=True)
        self.assertEqual(sorted(name for name, _ in inputs), ["c", "p", "x", "y"])
        self.assertEqual(sorted(name for name, _ in outputs), ["f", "g", "h", "z"])

    def test_baseline_values_copied_from_server(self):
        assert_near_equal(self.prob.get_val("x"), np.array([1.0, 2.0]))
        assert_near_equal(self.prob.get_val("y"), 3.0)
        assert_near_equal(self.prob.get_val("p"), 0.5)
        assert_near_equal(self.prob.get_val("c"), 10.0)

    def test_units_copied_from_server(self):
        meta = self.remote.get_io_metadata(metadata_keys=["units"])
        self.assertEqual(meta["x"]["units"], "m")
        self.assertEqual(meta["p"]["units"], "kg")
        self.assertIsNone(meta["y"]["units"])
        self.assertIsNone(meta["c"]["units"])

    def test_server_sends_unscaled_bounds(self):
        # the values the client receives must be in model units regardless of
        # how the server's OpenMDAO version stores bounds internally
        sent = self.remote.output_dict
        assert_near_equal(sent["design_vars"]["x"]["lower"], -10.0)
        assert_near_equal(sent["design_vars"]["x"]["upper"], 10.0)
        assert_near_equal(sent["design_vars"]["y"]["lower"], -5.0)
        assert_near_equal(sent["design_vars"]["y"]["upper"], 5.0)
        assert_near_equal(sent["constraints"]["g"]["upper"], 20.0)
        assert_near_equal(sent["constraints"]["h"]["equals"], 1.0)
        self.assertIsNone(sent["constraints"]["g"]["equals"])
        self.assertFalse(self.remote._lower_bound_used(sent["constraints"]["g"]["lower"]))

    def test_design_vars_match_server(self):
        dvs = self.prob.model.get_design_vars()
        server_dvs = self.server_model.get_design_vars()
        self.assertEqual(sorted(dvs.keys()), ["x", "y"])
        keys = ["lower", "upper", "ref", "ref0", "scaler", "adder"]
        self._assert_meta_matches_server(dvs["x"], server_dvs["x"], keys)
        self._assert_meta_matches_server(dvs["y"], server_dvs["y"], keys)

    def test_objective_matches_server(self):
        objs = self.prob.model.get_objectives()
        server_objs = self.server_model.get_objectives()
        self.assertEqual(list(objs.keys()), ["f"])
        self._assert_meta_matches_server(
            objs["f"], server_objs["f"], ["ref", "ref0", "scaler", "adder"]
        )

    def test_constraints_match_server(self):
        cons = self.prob.model.get_constraints()
        server_cons = self.server_model.get_constraints()
        self.assertEqual(sorted(cons.keys()), ["g", "h"])
        keys = ["lower", "upper", "equals", "ref", "ref0", "scaler", "adder"]
        self._assert_meta_matches_server(cons["g"], server_cons["g"], keys)
        self._assert_meta_matches_server(cons["h"], server_cons["h"], keys)
        self.assertIsNone(cons["g"]["equals"])
        self.assertIsNotNone(cons["h"]["equals"])

    def test_initialize_only_message_sent_during_setup(self):
        self.assertEqual(self.remote.server.messages_received, ["initialize"])


class TestRemoteCompEvaluation(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        self.prob = build_problem()
        self.prob.setup()
        self.remote = self.prob.model.remote

    def test_run_model_baseline(self):
        self.prob.run_model()
        expected = expected_outputs([1.0, 2.0], 3.0, 0.5, 10.0)
        for name, val in expected.items():
            assert_near_equal(self.prob.get_val(name), val, tolerance=1e-12)

    def test_run_model_new_design(self):
        self.prob.set_val("x", np.array([-2.0, 0.5]))
        self.prob.set_val("y", 1.5)
        self.prob.set_val("p", -1.0)
        self.prob.set_val("c", 2.0)
        self.prob.run_model()
        expected = expected_outputs([-2.0, 0.5], 1.5, -1.0, 2.0)
        for name, val in expected.items():
            assert_near_equal(self.prob.get_val(name), val, tolerance=1e-12)

    def test_server_skips_reevaluation_of_same_design(self):
        self.prob.run_model()
        counter_after_first = self.remote.output_dict["design_counter"]
        self.prob.run_model()
        self.assertEqual(
            self.remote.server.design_counter,
            counter_after_first,
        )
        self.prob.set_val("y", 4.0)
        self.prob.run_model()
        self.assertEqual(self.remote.server.design_counter, counter_after_first + 1)

    def test_compute_totals(self):
        self.prob.set_val("x", np.array([0.3, -1.2]))
        self.prob.set_val("y", 0.7)
        self.prob.run_model()
        totals = self.prob.compute_totals(of=["f", "g", "h", "z"], wrt=["x", "y", "p"])
        x = np.array([0.3, -1.2])
        y = 0.7
        assert_near_equal(totals["f", "x"], 2 * x.reshape(1, 2), tolerance=1e-12)
        assert_near_equal(totals["f", "y"], [[2 * y]], tolerance=1e-12)
        assert_near_equal(totals["f", "p"], [[1.0]], tolerance=1e-12)
        assert_near_equal(totals["g", "x"], [[1.0, 1.0]], tolerance=1e-12)
        assert_near_equal(totals["g", "y"], [[0.0]], tolerance=1e-12)
        assert_near_equal(totals["h", "x"], [[-1.0, 0.0]], tolerance=1e-12)
        assert_near_equal(totals["h", "y"], [[1.0]], tolerance=1e-12)
        assert_near_equal(totals["z", "x"], 3 * np.eye(2), tolerance=1e-12)
        assert_near_equal(totals["z", "y"], [[1.0], [1.0]], tolerance=1e-12)
        assert_near_equal(totals["z", "p"], [[0.0], [0.0]], tolerance=1e-12)

    def test_check_partials(self):
        self.prob.run_model()
        partials = self.prob.check_partials(compact_print=True, out_stream=None)
        checked = 0
        for (of, wrt), data in partials["remote"].items():
            if wrt == "c":  # constants are not differentiated on the server
                continue
            assert_near_equal(data["abs error"].forward, 0.0, tolerance=1e-5)
            checked += 1
        self.assertEqual(checked, 12)

    def test_server_replies_to_ping_without_evaluating(self):
        server = self.remote.server
        counter = server.design_counter
        reply = server.handle("ping|null")
        self.assertEqual(json.loads(reply), Server.PING_REPLY)
        self.assertEqual(server.design_counter, counter)
        self.assertEqual(server.messages_received[-1], "ping")
        self.prob.run_model()
        assert_near_equal(self.prob.get_val("f"), 14.5, tolerance=1e-12)

    def test_command_sequence(self):
        self.prob.run_model()
        self.prob.compute_totals(of=["f"], wrt=["x"])
        self.assertEqual(
            self.remote.server.messages_received,
            ["initialize", "evaluate", "evaluate derivatives"],
        )


class TestRemoteCompOptions(unittest.TestCase):
    N_PROCS = 1

    def test_dot_replacement_in_variable_names(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "remote",
            InProcessRemoteComp(
                server_factory=lambda: InProcessServer(get_nested_group),
                var_naming_dot_replacement="__",
            ),
            promotes=["*"],
        )
        prob.setup()
        prob.run_model()
        assert_near_equal(prob.get_val("sub__x"), 2.0)
        assert_near_equal(prob.get_val("sub__f"), 4.0)
        self.assertIn("sub__x", prob.model.get_design_vars())
        self.assertIn("sub__f", prob.model.get_objectives())
        totals = prob.compute_totals(of=["sub__f"], wrt=["sub__x"])
        assert_near_equal(totals["sub__f", "sub__x"], [[4.0]], tolerance=1e-12)

    def test_skip_objective_constraint_definition(self):
        prob = build_problem(skip_objective_constraint_definition=True)
        prob.setup()
        prob.final_setup()
        self.assertEqual(list(prob.model.get_design_vars().keys()), ["x", "y"])
        self.assertEqual(prob.model.get_objectives(), {})
        self.assertEqual(prob.model.get_constraints(), {})
        prob.run_model()
        assert_near_equal(prob.get_val("f"), 14.5)

    def test_restart_when_time_is_not_enough(self):
        manager = RecordingServerManager(time_remaining=False)
        prob = build_problem(server_manager=manager)
        prob.setup()
        self.assertEqual((manager.start_calls, manager.stop_calls), (0, 0))
        prob.run_model()
        self.assertEqual((manager.start_calls, manager.stop_calls), (1, 1))
        prob.compute_totals(of=["f"], wrt=["x"])
        # reboot_only_on_function_call defaults to True
        self.assertEqual((manager.start_calls, manager.stop_calls), (1, 1))
        prob.set_val("y", 1.0)
        prob.run_model()
        self.assertEqual((manager.start_calls, manager.stop_calls), (2, 2))
        assert_near_equal(prob.get_val("f"), 1.0 + 4.0 + 1.0 + 0.5, tolerance=1e-12)

    def test_restart_before_derivatives_when_allowed(self):
        manager = RecordingServerManager(time_remaining=False)
        prob = build_problem(
            server_manager=manager, reboot_only_on_function_call=False
        )
        prob.setup()
        prob.run_model()
        self.assertEqual((manager.start_calls, manager.stop_calls), (1, 1))
        prob.compute_totals(of=["f"], wrt=["x"])
        # first gradient evaluation never triggers a restart
        self.assertEqual((manager.start_calls, manager.stop_calls), (1, 1))
        prob.set_val("y", 1.0)
        prob.compute_totals(of=["f"], wrt=["x"])
        self.assertEqual((manager.start_calls, manager.stop_calls), (2, 2))

    def test_no_restart_when_time_is_enough(self):
        manager = RecordingServerManager(time_remaining=True)
        prob = build_problem(server_manager=manager)
        prob.setup()
        prob.run_model()
        prob.compute_totals(of=["f"], wrt=["x"])
        prob.set_val("y", 1.0)
        prob.run_model()
        self.assertEqual((manager.start_calls, manager.stop_calls), (0, 0))

    def test_stop_server_for_down_time_after_every_call(self):
        manager = RecordingServerManager()
        prob = build_problem(server_manager=manager, stop_server_for_down_time=1)
        prob.setup()
        self.assertTrue(manager.stopped)
        self.assertEqual(manager.stop_calls, 1)
        prob.run_model()
        self.assertTrue(manager.stopped)
        self.assertEqual(manager.start_calls, 1)
        assert_near_equal(prob.get_val("f"), 14.5)

    def test_stop_server_for_down_time_after_derivatives_only(self):
        manager = RecordingServerManager()
        prob = build_problem(server_manager=manager, stop_server_for_down_time=2)
        prob.setup()
        prob.run_model()
        self.assertFalse(manager.stopped)
        self.assertEqual(manager.stop_calls, 0)
        prob.compute_totals(of=["f"], wrt=["x"])
        self.assertTrue(manager.stopped)
        self.assertEqual(manager.stop_calls, 1)
        prob.set_val("y", 1.0)
        prob.run_model()
        self.assertEqual(manager.start_calls, 1)
        self.assertFalse(manager.stopped)

    def test_nan_design_var_raises_and_stops_server(self):
        manager = RecordingServerManager()
        prob = build_problem(server_manager=manager)
        prob.setup()
        server = prob.model.remote.server
        counter = server.design_counter
        prob.set_val("x", np.array([1.0, np.nan]))
        with self.assertRaisesRegex(ValueError, r"NaN found in inputs.*\bx\b"):
            prob.run_model()
        self.assertEqual(manager.stop_calls, 1)
        self.assertTrue(manager.stopped)
        self.assertEqual(server.design_counter, counter)
        self.assertEqual(server.messages_received, ["initialize"])

    def test_nan_additional_input_raises_in_compute_partials(self):
        manager = RecordingServerManager()
        prob = build_problem(server_manager=manager)
        prob.setup()
        prob.run_model()
        # set directly on the component's input vector: compute_totals does not
        # re-transfer inputs, so a set_val would not reach compute_partials
        prob.model.remote._inputs["p"] = np.nan
        with self.assertRaisesRegex(ValueError, r"NaN found in inputs.*\bp\b"):
            prob.compute_totals(of=["f"], wrt=["x"])
        self.assertEqual(manager.stop_calls, 1)
        self.assertNotIn("evaluate derivatives", prob.model.remote.server.messages_received)

    def test_nan_in_multiple_inputs_lists_all(self):
        prob = build_problem(server_manager=RecordingServerManager())
        prob.setup()
        prob.set_val("y", np.nan)
        prob.set_val("p", np.nan)
        with self.assertRaisesRegex(ValueError, r"\by\b.*\bp\b|\bp\b.*\by\b"):
            prob.run_model()

    def test_nan_constant_is_not_checked(self):
        # constants are not part of the optimization; leave validation to the server model
        prob = build_problem(server_manager=RecordingServerManager())
        prob.setup()
        prob.set_val("c", np.nan)
        prob.run_model()
        self.assertTrue(np.isnan(prob.get_val("g")))

    def test_top_level_stop_server_shortcut(self):
        manager = RecordingServerManager()
        prob = build_problem(server_manager=manager)
        prob.setup()
        prob.model.remote.stop_server()
        self.assertEqual(manager.stop_calls, 1)

    def test_stop_server_without_manager_is_noop(self):
        comp = InProcessRemoteComp(server_factory=UnreachableServer)
        comp.stop_server()


class TestRemoteCompJsonDump(unittest.TestCase):
    N_PROCS = 1

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dump_json_single_file(self):
        prob = build_problem(dump_json=True, run_directory=self.run_dir)
        prob.setup()
        prob.run_model()
        for kind in ["inputs", "outputs"]:
            path = os.path.join(self.run_dir, f"remote_{kind}.json")
            self.assertTrue(os.path.isfile(path), path)
        with open(os.path.join(self.run_dir, "remote_outputs.json")) as f:
            dumped = json.load(f)
        self.assertIn("wall_time", dumped)
        self.assertIn("down_time", dumped)
        assert_near_equal(dumped["objective"]["f"]["val"], [14.5])

    def test_dump_separate_json_files(self):
        prob = build_problem(dump_separate_json=True, run_directory=self.run_dir)
        prob.setup()
        prob.run_model()
        prob.compute_totals(of=["f"], wrt=["x"])
        prob.set_val("y", 1.0)
        prob.run_model()
        json_dir = os.path.join(self.run_dir, "remote_json_files")
        expected_files = {
            "remote_inputs_function0.json",
            "remote_outputs_function0.json",
            "remote_inputs_function1.json",
            "remote_outputs_function1.json",
            "remote_inputs_derivative0.json",
            "remote_outputs_derivative0.json",
            "remote_inputs_function2.json",
            "remote_outputs_function2.json",
        }
        self.assertEqual(set(os.listdir(json_dir)), expected_files)
        with open(os.path.join(json_dir, "remote_outputs_derivative0.json")) as f:
            dumped = json.load(f)
        self.assertIn("derivatives", dumped["objective"]["f"])

    def test_reuse_dumped_json_avoids_server(self):
        prob = build_problem(dump_separate_json=True, run_directory=self.run_dir)
        prob.setup()
        prob.run_model()
        prob.compute_totals(of=["f", "g", "h", "z"], wrt=["x", "y", "p"])
        prob.set_val("x", np.array([0.3, -1.2]))
        prob.run_model()
        totals_ref = prob.compute_totals(of=["f", "g", "h", "z"], wrt=["x", "y", "p"])

        manager = RecordingServerManager()
        prob2 = build_problem(
            server_factory=UnreachableServer,
            server_manager=manager,
            reuse_dumped_json=True,
            run_directory=self.run_dir,
        )
        prob2.setup()
        prob2.run_model()
        assert_near_equal(prob2.get_val("f"), 14.5, tolerance=1e-12)
        prob2.set_val("x", np.array([0.3, -1.2]))
        prob2.run_model()
        expected = expected_outputs([0.3, -1.2], 3.0, 0.5, 10.0)
        for name, val in expected.items():
            assert_near_equal(prob2.get_val(name), val, tolerance=1e-12)
        totals = prob2.compute_totals(of=["f", "g", "h", "z"], wrt=["x", "y", "p"])
        for key in totals_ref:
            assert_near_equal(totals[key], totals_ref[key], tolerance=1e-12)
        self.assertIsNone(prob2.model.remote.server)

    def test_reuse_dumped_json_falls_back_to_server_for_new_design(self):
        prob = build_problem(dump_separate_json=True, run_directory=self.run_dir)
        prob.setup()
        prob.run_model()

        prob2 = build_problem(reuse_dumped_json=True, run_directory=self.run_dir)
        prob2.setup()
        self.assertIsNone(prob2.model.remote.server)
        prob2.set_val("y", -2.0)
        prob2.run_model()
        self.assertEqual(prob2.model.remote.server.messages_received, ["evaluate"])
        expected = expected_outputs([1.0, 2.0], -2.0, 0.5, 10.0)
        for name, val in expected.items():
            assert_near_equal(prob2.get_val(name), val, tolerance=1e-12)


@unittest.skipUnless(
    MPI.COMM_WORLD.size == 2,
    "requires 2 MPI processes (run with testflo, or mpiexec -n 2 python -m unittest)",
)
class TestRemoteCompParallel(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.prob = build_problem()
        self.prob.setup()
        self.remote = self.prob.model.remote

    def test_server_only_created_on_root(self):
        if self.comm.rank == 0:
            self.assertIsNotNone(self.remote.server)
            self.assertIsNotNone(self.remote.server_manager)
        else:
            self.assertIsNone(self.remote.server)
            self.assertIsNone(self.remote.server_manager)

    def test_outputs_broadcast_to_all_ranks(self):
        self.prob.set_val("x", np.array([-2.0, 0.5]))
        self.prob.set_val("y", 1.5)
        self.prob.run_model()
        expected = expected_outputs([-2.0, 0.5], 1.5, 0.5, 10.0)
        for name, val in expected.items():
            assert_near_equal(self.prob.get_val(name), val, tolerance=1e-12)
        gathered = self.comm.allgather(self.prob.get_val("z"))
        self.assertEqual(len(gathered), self.comm.size)
        for z_on_rank in gathered[1:]:
            assert_near_equal(z_on_rank, gathered[0], tolerance=1e-12)

    def test_totals_broadcast_to_all_ranks(self):
        self.prob.run_model()
        totals = self.prob.compute_totals(of=["f"], wrt=["x"])
        assert_near_equal(totals["f", "x"], [[2.0, 4.0]], tolerance=1e-12)

    def test_nan_input_raises_on_all_ranks(self):
        self.prob.set_val("y", np.nan)
        with self.assertRaisesRegex(ValueError, "NaN found in inputs"):
            self.prob.run_model()
        if self.comm.rank == 0:
            self.assertTrue(self.remote.server_manager.stopped)


if __name__ == "__main__":
    unittest.main()
