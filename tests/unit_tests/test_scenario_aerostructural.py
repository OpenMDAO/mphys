import unittest

import openmdao.api as om
from mpi4py import MPI

from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.coupling_aerostructural import CouplingAeroStructural
from mphys.geo_disp import GeoDisp

from common_methods import CommonMethods
from fake_aero import AeroBuilder, AeroMeshComp, AeroPreCouplingComp, AeroCouplingComp, AeroPostCouplingComp
from fake_struct import StructBuilder, StructMeshComp, StructPreCouplingComp, StructCouplingComp, StructPostCouplingComp
from fake_geometry import GeometryBuilder, Geometry
from fake_ldxfer import LDXferBuilder, LoadXferComp, DispXferComp

class TestScenarioAeroStructural(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()

        aero_builder = AeroBuilder()
        struct_builder = StructBuilder()
        ldxfer_builder = LDXferBuilder(aero_builder, struct_builder)

        aero_builder.initialize(MPI.COMM_WORLD)
        struct_builder.initialize(MPI.COMM_WORLD)
        ldxfer_builder.initialize(MPI.COMM_WORLD)

        self.prob.model.add_subsystem('aero_mesh', aero_builder.get_mesh_coordinate_subsystem())
        self.prob.model.add_subsystem('struct_mesh', struct_builder.get_mesh_coordinate_subsystem())
        self.prob.model.add_subsystem('scenario', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                  struct_builder=struct_builder,
                                                                  ldxfer_builder=ldxfer_builder))
        self.prob.model.connect('aero_mesh.x_aero0', 'scenario.x_aero0')
        self.prob.model.connect('struct_mesh.x_struct0', 'scenario.x_struct0')
        self.prob.setup()

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_scenario_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.aero_pre, AeroPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_pre, StructPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingAeroStructural)
        self.assertIsInstance(self.prob.model.scenario.aero_post, AeroPostCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_post, StructPostCouplingComp)

    def test_scenario_subsystem_order(self):
        expected_order = ['aero_pre', 'struct_pre', 'coupling', 'aero_post', 'struct_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)

    def test_coupling_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.coupling.aero, AeroCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.struct, StructCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.disp_xfer, DispXferComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.load_xfer, LoadXferComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.geo_disp, GeoDisp)

    def test_coupling_subsystem_order(self):
        expected_order = ['disp_xfer', 'geo_disp', 'aero', 'load_xfer', 'struct']
        self.common.test_subsystem_order(self, self.prob.model.scenario.coupling, expected_order)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)


class TestScenarioAeroStructuralParallel(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()

        aero_builder = AeroBuilder()
        struct_builder = StructBuilder()
        ldxfer_builder = LDXferBuilder(aero_builder, struct_builder)

        self.prob.model.add_subsystem('scenario', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                  struct_builder=struct_builder,
                                                                  ldxfer_builder=ldxfer_builder,
                                                                  in_MultipointParallel=True))
        self.prob.setup()

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_scenario_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.aero_mesh, AeroMeshComp)
        self.assertIsInstance(self.prob.model.scenario.struct_mesh, StructMeshComp)
        self.assertIsInstance(self.prob.model.scenario.aero_pre, AeroPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_pre, StructPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingAeroStructural)
        self.assertIsInstance(self.prob.model.scenario.aero_post, AeroPostCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_post, StructPostCouplingComp)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_scenario_subsystem_order(self):
        expected_order = ['aero_mesh', 'struct_mesh', 'aero_pre', 'struct_pre',
                          'coupling', 'aero_post', 'struct_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)

    def test_coupling_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.coupling.aero, AeroCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.struct, StructCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.disp_xfer, DispXferComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.load_xfer, LoadXferComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.geo_disp, GeoDisp)

    def test_coupling_subsystem_order(self):
        expected_order = ['disp_xfer', 'geo_disp', 'aero', 'load_xfer', 'struct']
        self.common.test_subsystem_order(self, self.prob.model.scenario.coupling, expected_order)


class TestScenarioAeroStructuralParallelWithGeometry(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()

        aero_builder = AeroBuilder()
        struct_builder = StructBuilder()
        ldxfer_builder = LDXferBuilder(aero_builder, struct_builder)
        geometry_builder = GeometryBuilder(['aero','struct'], [aero_builder, struct_builder])

        self.prob.model.add_subsystem('scenario', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                  struct_builder=struct_builder,
                                                                  ldxfer_builder=ldxfer_builder,
                                                                  geometry_builder=geometry_builder,
                                                                  in_MultipointParallel=True))
        self.prob.setup()

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_scenario_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.aero_mesh, AeroMeshComp)
        self.assertIsInstance(self.prob.model.scenario.struct_mesh, StructMeshComp)
        self.assertIsInstance(self.prob.model.scenario.geometry, Geometry)
        self.assertIsInstance(self.prob.model.scenario.aero_pre, AeroPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_pre, StructPreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingAeroStructural)
        self.assertIsInstance(self.prob.model.scenario.aero_post, AeroPostCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.struct_post, StructPostCouplingComp)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_scenario_subsystem_order(self):
        expected_order = ['aero_mesh', 'struct_mesh', 'geometry', 'aero_pre', 'struct_pre',
                          'coupling', 'aero_post', 'struct_post']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)

    def test_coupling_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.coupling.aero, AeroCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.struct, StructCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.disp_xfer, DispXferComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.load_xfer, LoadXferComp)
        self.assertIsInstance(self.prob.model.scenario.coupling.geo_disp, GeoDisp)

    def test_coupling_subsystem_order(self):
        expected_order = ['disp_xfer', 'geo_disp', 'aero', 'load_xfer', 'struct']
        self.common.test_subsystem_order(self, self.prob.model.scenario.coupling, expected_order)


if __name__ == '__main__':
    unittest.main()
