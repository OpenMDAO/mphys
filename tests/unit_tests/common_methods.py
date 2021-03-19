"""
Have a single implementation of tests that many unit tests will run
"""


def test_run_model(self):
    self.prob.run_model()


def test_no_autoivcs(self):
    for output in self.prob.model._conn_global_abs_in2out.values():
        self.assertFalse('auto_ivc' in output)
