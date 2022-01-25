import unittest
import numpy as np

import openmdao.api as om
from mpi4py import MPI

from mphys import Builder
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.coupling_aerostructural import CouplingAeroStructural
from mphys.geo_disp import GeoDisp
from common_methods import CommonMethods

num_nodes = 3


class AeroMeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output('x_aero0', val=np.ones(num_nodes*3), tags=['mphys_coordinates'])


class AeroPreCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('x_aero0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output('prestate_aero', tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['prestate_aero'] = np.sum(inputs['x_aero0'])


class AeroCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('prestate_aero', tags=['mphys_coupling'])
        self.add_output('f_aero', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['f_aero'] = inputs['x_aero'] + inputs['prestate_aero']


class AeroPostCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('prestate_aero', tags=['mphys_coupling'])
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('f_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('func_aero', val=1.0, tags=['mphys_result'])

    def compute(self, inputs, outputs):
        outputs['func_aero'] = np.sum(inputs['f_aero'] + inputs['prestate_aero'] + inputs['x_aero'])


class AeroBuilder(Builder):
    def get_number_of_nodes(self):
        return num_nodes

    def get_ndof(self):
        return 3

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return AeroMeshComp()

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return AeroPreCouplingComp()

    def get_coupling_group_subsystem(self, scenario_name=None):
        return AeroCouplingComp()

    def get_post_coupling_subsystem(self, scenario_name=None):
        return AeroPostCouplingComp()


class StructMeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output('x_struct0', val=np.ones(num_nodes*3), tags=['mphys_coordinates'])


class StructPreCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output('prestate_struct', tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['prestate_struct'] = np.sum(inputs['x_struct0'])


class StructCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('prestate_struct', tags=['mphys_coupling'])
        self.add_input('f_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('u_struct', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['u_struct'] = inputs['x_struct0'] + inputs['prestate_struct']


class StructPostCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('prestate_struct', tags=['mphys_coupling'])
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('func_struct', val=1.0, tags=['mphys_result'])

    def compute(self, inputs, outputs):
        outputs['func_struct'] = np.sum(
            inputs['u_struct'] + inputs['prestate_struct'] + inputs['x_struct0'])


class StructBuilder(Builder):
    def get_number_of_nodes(self):
        return num_nodes

    def get_ndof(self):
        return 3

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return StructMeshComp()

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return StructPreCouplingComp()

    def get_coupling_group_subsystem(self, scenario_name=None):
        return StructCouplingComp()

    def get_post_coupling_subsystem(self, scenario_name=None):
        return StructPostCouplingComp()


class DispXferComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('u_aero', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['u_aero'] = inputs['u_struct']


class LoadXferComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('f_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('f_struct', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['f_struct'] = inputs['f_aero']


class LDXferBuilder(Builder):
    def get_number_of_nodes(self):
        return num_nodes

    def get_ndof(self):
        return 3

    def get_coupling_group_subsystem(self, scenario_name=None):
        return DispXferComp(), LoadXferComp()


class Geometry(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_aero_in', shape_by_conn=True)
        self.add_output('x_aero0', shape=3*num_nodes, tags=['mphys_coordinates'])

        self.add_input('x_struct_in', shape_by_conn=True)
        self.add_output('x_struct0', shape=3*num_nodes, tags=['mphys_coordinates'])

    def compute(self, inputs, outputs):
        outputs['x_aero0'] = inputs['x_aero_in']
        outputs['x_struct0'] = inputs['x_struct_in']


class GeometryBuilder(Builder):
    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return Geometry()


class TestScenarioAeroStructural(unittest.TestCase):
    def setUp(self):
        self.common = CommonMethods()
        self.prob = om.Problem()

        aero_builder = AeroBuilder()
        struct_builder = StructBuilder()
        ldxfer_builder = LDXferBuilder()

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
        ldxfer_builder = LDXferBuilder()

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
        ldxfer_builder = LDXferBuilder()
        geometry_builder = GeometryBuilder()

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
