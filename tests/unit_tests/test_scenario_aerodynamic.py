import unittest
import numpy as np

import openmdao.api as om
from mpi4py import MPI

from mphys import Builder
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from common_methods import CommonMethods
from fake_aero import AeroBuilder, AeroMeshComp, AeroPreCouplingComp, AeroCouplingComp, AeroPostCouplingComp
from fake_geometry import Geometry, GeometryBuilder


class TestScenarioAerodynamic(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = AeroBuilder()
        builder.initialize(MPI.COMM_WORLD)
        self.prob.model.add_subsystem('mesh', builder.get_mesh_coordinate_subsystem())
        self.prob.model.add_subsystem('scenario', ScenarioAerodynamic(aero_builder=builder))
        self.prob.model.connect('mesh.x_aero0', 'scenario.x_aero')
        self.prob.setup()

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.aero_pre, AeroPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, AeroCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.aero_post, AeroPostCouplingComp)

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['aero_pre', 'coupling', 'aero_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)


class TestScenarioAerodynamicParallel(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = AeroBuilder()
        self.prob.model.add_subsystem('scenario', ScenarioAerodynamic(aero_builder=builder,
                                                                      in_MultipointParallel=True))
        self.prob.setup()

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.mesh, AeroMeshComp)
        self.assertIsInstance(self.prob.model.scenario.aero_pre, AeroPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, AeroCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.aero_post, AeroPostCouplingComp)

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['mesh', 'aero_pre', 'coupling', 'aero_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)


class TestScenarioAerodynamicParallelWithGeometry(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = AeroBuilder()
        geom_builder = GeometryBuilder(['aero'], [builder])
        self.prob.model.add_subsystem('scenario', ScenarioAerodynamic(aero_builder=builder,
                                                                      geometry_builder=geom_builder,
                                                                      in_MultipointParallel=True))
        self.prob.setup()

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.mesh, AeroMeshComp)
        self.assertIsInstance(self.prob.model.scenario.geometry, Geometry)
        self.assertIsInstance(self.prob.model.scenario.aero_pre, AeroPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, AeroCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.aero_post, AeroPostCouplingComp)

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_subsystem_order(self):
        expected_order = ['mesh', 'geometry', 'aero_pre', 'coupling', 'aero_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)

    def test_coordinates_connected_from_geometry(self):
        scenario = self.prob.model.scenario
        systems = ['aero_pre', 'coupling', 'aero_post']
        for sys in systems:
            self.assertEqual(scenario._conn_global_abs_in2out[f'scenario.{sys}.x_aero'],
                             'scenario.geometry.x_aero0')


if __name__ == '__main__':
    unittest.main()
