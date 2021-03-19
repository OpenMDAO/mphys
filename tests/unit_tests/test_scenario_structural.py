import unittest
import numpy as np

import openmdao.api as om
from mpi4py import MPI

from mphys import Builder
from mphys.scenario_structural import ScenarioStructural
from common_methods import CommonMethods

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


class Geometry(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_struct_in', shape_by_conn=True)
        self.add_output('x_struct0', shape=3*num_nodes, tags=['mphys_coordinates'])

    def compute(self, inputs, outputs):
        outputs['x_struct0'] = inputs['x_struct_in']


class GeometryBuilder(Builder):
    def get_mesh_coordinate_subsystem(self):
        return Geometry()


class TestScenarioStructural(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = StructBuilder()
        builder.initialize(MPI.COMM_WORLD)
        self.prob.model.add_subsystem('mesh', builder.get_mesh_coordinate_subsystem())
        self.prob.model.add_subsystem('scenario', ScenarioStructural(struct_builder=builder))
        self.prob.model.connect('mesh.x_struct0', 'scenario.x_struct0')
        self.prob.setup()
        om.n2(self.prob, outfile='n2/test_scenario_structural.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_post, PostCouplingComp)

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['struct_pre','coupling','struct_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)


class TestScenarioStructuralParallel(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
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
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['mesh', 'struct_pre','coupling','struct_post']
        self.common.test_subsystem_order(self, self.prob.model, expected_order)


class TestScenarioStructuralParallelWithGeometry(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = StructBuilder()
        geom_builder = GeometryBuilder()
        self.prob.model = ScenarioStructural(struct_builder=builder, geometry_builder=geom_builder,
                                             in_MultipointParallel=True)
        self.prob.setup()
        om.n2(self.prob, outfile='n2/test_scenario_structural_parallel_geometry.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.mesh, MeshComp)
        self.assertIsInstance(self.prob.model.geometry, Geometry)
        self.assertIsInstance(self.prob.model.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.struct_post, PostCouplingComp)

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['mesh', 'geometry', 'struct_pre','coupling','struct_post']
        self.common.test_subsystem_order(self, self.prob.model, expected_order)


if __name__ == '__main__':
    unittest.main()
