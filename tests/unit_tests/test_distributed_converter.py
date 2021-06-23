import unittest
from mpi4py import MPI
import numpy as np
import openmdao.api as om

from mphys import DistributedConverter, DistributedVariableDescription

from common_methods import CommonMethods
from openmdao.utils.assert_utils import assert_near_equal

class TestDistributedConverter(unittest.TestCase):
    N_PROCS = 1 #TODO should be 2 or more but there is a bug in OM currently

    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()

        vars_in = [
            DistributedVariableDescription('in1', shape=(4, 4), tags=['mphys_coupling']),
            DistributedVariableDescription('in2', shape=(10), tags=['mphys_coordinates'])
        ]

        vars_out = [
            DistributedVariableDescription('out1', shape=(5), tags=['mphys_coupling']),
            DistributedVariableDescription('out2', shape=(15), tags=['mphys_result'])
        ]

        inputs = self.prob.model.add_subsystem('inputs', om.IndepVarComp())

        in1_shape = (4, 4) if MPI.COMM_WORLD.Get_rank() == 0 else 0
        in2_shape = 10 if MPI.COMM_WORLD.Get_rank() == 0 else 0
        inputs.add_output('in1', val=np.ones(in1_shape, dtype=float), distributed=True)
        inputs.add_output('in2', val=np.arange(in2_shape, dtype=float), distributed=True)

        out1_shape = 5
        out2_shape = 15
        inputs.add_output('out1', val=np.ones(out1_shape, dtype=float), distributed=False)
        inputs.add_output('out2', val=np.arange(out2_shape, dtype=float), distributed=False)

        self.prob.model.add_subsystem('converter', DistributedConverter(
            distributed_inputs=vars_in, distributed_outputs=vars_out))

        for var in ['in1', 'in2']:
            self.prob.model.connect(f'inputs.{var}', f'converter.{var}')
        for var in ['out1', 'out2']:
            self.prob.model.connect(f'inputs.{var}', f'converter.{var}_serial')

        self.prob.setup(force_alloc_complex=True)

    def test_run_model(self):
        self.common.test_run_model(self, write_n2=False)

    def test_check_partials(self):
        partials = self.prob.check_partials(compact_print=True, method='cs')
        tol= 1e-9
        for in_var in ['in1','in2']:
            rel_error = partials['converter'][(f'{in_var}_serial',in_var)]['rel error']
            assert_near_equal(rel_error.reverse, 0.0, tolerance = tol)
            assert_near_equal(rel_error.forward, 0.0, tolerance = tol)
            assert_near_equal(rel_error.forward_reverse, 0.0, tolerance = tol)
        for out_var in ['out1','out2']:
            rel_error = partials['converter'][(out_var,f'{out_var}_serial')]['rel error']
            assert_near_equal(rel_error.reverse, 0.0, tolerance = tol)
            assert_near_equal(rel_error.forward, 0.0, tolerance = tol)
            assert_near_equal(rel_error.forward_reverse, 0.0, tolerance = tol)



if __name__ == '__main__':
    unittest.main()
