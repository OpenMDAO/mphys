import unittest
import numpy as np

import openmdao.api as om
from mpi4py import MPI

from mphys import MPhysVariables
from mphys.scenarios import ScenarioStructural
from common_methods import CommonMethods

from fake_struct import StructBuilder, StructMeshComp, StructCouplingComp, StructPostCouplingComp, struct_num_nodes
from fake_geometry import Geometry, GeometryBuilder


class PreCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.x_struct0_name = MPhysVariables.Structures.COORDINATES
        self.f_struct_name = MPhysVariables.Structures.LOADS_FROM_AERODYNAMICS
        self.add_input(self.x_struct0_name, shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output(self.f_struct_name, val=np.ones(struct_num_nodes*3), tags=['mphys_coupling'])
        self.add_output('prestate_struct', tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs[self.f_struct_name] = inputs[self.x_struct0_name]
        outputs['prestate_struct'] = np.sum(inputs[self.x_struct0_name])

class FakeStructBuilderWithLoads(StructBuilder):
    def get_pre_coupling_subsystem(self, scenario_name=None):
        return PreCouplingComp()

class TestScenarioStructural(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = FakeStructBuilderWithLoads()
        builder.initialize(MPI.COMM_WORLD)
        self.prob.model.add_subsystem('mesh', builder.get_mesh_coordinate_subsystem())
        self.prob.model.add_subsystem('scenario', ScenarioStructural(struct_builder=builder))
        self.prob.model.connect(f'mesh.{MPhysVariables.Structures.Mesh.COORDINATES}',
                                f'scenario.{MPhysVariables.Structures.COORDINATES}')

        self.prob.setup()

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, StructCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_post, StructPostCouplingComp)

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
        builder = FakeStructBuilderWithLoads()
        self.prob.model = ScenarioStructural(struct_builder=builder, in_MultipointParallel=True)
        self.prob.setup()

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.mesh, StructMeshComp)
        self.assertIsInstance(self.prob.model.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.coupling, StructCouplingComp)
        self.assertIsInstance(self.prob.model.struct_post, StructPostCouplingComp)

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
        builder = FakeStructBuilderWithLoads()
        geom_builder = GeometryBuilder(['struct'],[builder])
        self.prob.model = ScenarioStructural(struct_builder=builder, geometry_builder=geom_builder,
                                             in_MultipointParallel=True)
        self.prob.setup()

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.mesh, StructMeshComp)
        self.assertIsInstance(self.prob.model.geometry, Geometry)
        self.assertIsInstance(self.prob.model.struct_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.coupling, StructCouplingComp)
        self.assertIsInstance(self.prob.model.struct_post, StructPostCouplingComp)

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['mesh', 'geometry', 'struct_pre','coupling','struct_post']
        self.common.test_subsystem_order(self, self.prob.model, expected_order)


if __name__ == '__main__':
    unittest.main()
