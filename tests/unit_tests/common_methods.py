"""
Have a single implementation of tests that many unit tests will run
"""

class CommonMethods:

    def test_run_model(self, obj):
        obj.prob.run_model()

    def test_no_autoivcs(self, obj):
        for output in obj.prob.model._conn_global_abs_in2out.values():
            obj.assertFalse('auto_ivc' in output)
