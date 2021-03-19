import unittest
import numpy as np

import openmdao.api as om
from mpi4py import MPI

from mphys import Builder
from mphys.scenario_structural import ScenarioStructural
from common_methods import test_run_model, test_no_autoivcs

num_nodes = 3


class MeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output('x_struct0', val=np.ones(num_nodes*3), tags=['mphys_coordinates'])


class PreCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output('f_struct', val=np.ones(num_nodes*3), tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['f_struct'] = inputs['x_struct0']


class CouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('f_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('u_struct', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['u_struct'] = inputs['f_struct']


class PostCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('f_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('func_struct', val=1.0, tags=['mphys_result'])


class StructBuilder(Builder):
    def get_number_of_nodes(self):
        return num_nodes

    def get_ndof(self):
        return 3

    def get_mesh_coordinate_subsystem(self):
        return MeshComp()

    def get_pre_coupling_subsystem(self):
        return PreCouplingComp()

    def get_coupling_group_subsystem(self):
        return CouplingComp()

    def get_post_coupling_subsystem(self):
        return PostCouplingComp()


class TestScenarioStructural(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        builder = StructBuilder()
        builder.initialize(MPI.COMM_WORLD)
        self.prob.model.add_subsystem('mesh', builder.get_mesh_coordinate_subsystem())
        self.prob.model.add_subsystem('scen', ScenarioStructural(struct_builder=builder))
        self.prob.model.connect('mesh.x_struct0', 'scen.x_struct0')
        self.prob.setup()
        om.n2(self.prob, outfile='n2/test_scenario_structural.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scen.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.scen.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.scen.struct_post, PostCouplingComp)

    def test_run_model(self):
        test_run_model(self)

    def test_no_autoivcs(self):
        test_no_autoivcs(self)


class TestScenarioStructuralParallel(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        builder = StructBuilder()
        self.prob.model = ScenarioStructural(struct_builder=builder, in_MultipointParallel=True)
        self.prob.setup()
        om.n2(self.prob, outfile='n2/test_scenario_structural_parallel.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.mesh, MeshComp)
        self.assertIsInstance(self.prob.model.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.struct_post, PostCouplingComp)

    def test_run_model(self):
        test_run_model(self)

    def test_no_autoivcs(self):
        test_no_autoivcs(self)


if __name__ == '__main__':
    unittest.main()
