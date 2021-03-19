from typing import List
"""
Have a single implementation of tests that many unit tests will run
"""

class CommonMethods:

    def test_run_model(self, obj):
        obj.prob.run_model()

    def test_no_autoivcs(self, obj):
        for output in obj.prob.model._conn_global_abs_in2out.values():
            obj.assertFalse('auto_ivc' in output)

    def test_subsystem_order(self, obj, group, expected_order: List[str]):
        systems = group._subsystems_allprocs
        for name, subsystem in systems.items():
            for expected_index, expected_name in enumerate(expected_order):
                if name == expected_name:
                    obj.assertEqual(subsystem.index, expected_index)
                    break
            else:
                print('Unknown component', name)
                obj.assertTrue(False)
