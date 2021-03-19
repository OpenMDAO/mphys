import unittest
import numpy as np

import openmdao.api as om
from mpi4py import MPI

from mphys import Builder
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from common_methods import CommonMethods

num_nodes = 3


class MeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output('x_aero0', val=np.ones(num_nodes*3), tags=['mphys_coordinates'])


class PreCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output('aoa', tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['aoa'] = np.sum(inputs['x_aero'])


class CouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('aoa', tags=['mphys_coupling'])
        self.add_output('f_aero', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['f_aero'] = inputs['x_aero'] + inputs['aoa']


class PostCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('aoa', tags=['mphys_coupling'])
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('f_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('func_aero', val=1.0, tags=['mphys_result'])

    def compute(self, inputs, outputs):
        outputs['func_aero'] = np.sum(inputs['f_aero'] + inputs['aoa'] + inputs['x_aero'])


class AeroBuilder(Builder):
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
        self.add_input('x_aero_in', shape_by_conn=True)
        self.add_output('x_aero0', shape=3*num_nodes, tags=['mphys_coordinates'])

    def compute(self, inputs, outputs):
        outputs['x_aero0'] = inputs['x_aero_in']


class GeometryBuilder(Builder):
    def get_mesh_coordinate_subsystem(self):
        return Geometry()


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
        om.n2(self.prob, outfile=f'n2/{self.__class__.__name__}.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.aero_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.scenario.aero_post, PostCouplingComp)

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
        om.n2(self.prob, outfile=f'n2/{self.__class__.__name__}.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.mesh, MeshComp)
        self.assertIsInstance(self.prob.model.scenario.aero_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.scenario.aero_post, PostCouplingComp)

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
        geom_builder = GeometryBuilder()
        self.prob.model.add_subsystem('scenario', ScenarioAerodynamic(aero_builder=builder,
                                                                      geometry_builder=geom_builder,
                                                                      in_MultipointParallel=True))
        self.prob.setup()
        om.n2(self.prob, outfile=f'n2/{self.__class__.__name__}.html', show_browser=False)

    def test_mphys_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.mesh, MeshComp)
        self.assertIsInstance(self.prob.model.scenario.geometry, Geometry)
        self.assertIsInstance(self.prob.model.scenario.aero_pre, PreCouplingComp)
        self.assertIsInstance(self.prob.model.scenario.coupling, CouplingComp)
        self.assertIsInstance(self.prob.model.scenario.aero_post, PostCouplingComp)

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
