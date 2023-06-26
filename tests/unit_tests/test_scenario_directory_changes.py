from pathlib import Path
import shutil
import unittest
import numpy as np
import subprocess
import os

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
        self.add_output('prestate_aero', tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        print('TOUCH',self.name, os.getcwd())
        Path('precoupling_compute').touch()
        outputs['prestate_aero'] = np.sum(inputs['x_aero'])


class CouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('prestate_aero', tags=['mphys_coupling'])
        self.add_output('f_aero', shape=num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        print('TOUCH',self.name, os.getcwd())
        Path('coupling_compute').touch()
        outputs['f_aero'] = inputs['x_aero'] + inputs['prestate_aero']


class PostCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('prestate_aero', tags=['mphys_coupling'])
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('f_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('func_aero', val=1.0, tags=['mphys_result'])

    def compute(self, inputs, outputs):
        print('TOUCH',self.name, os.getcwd())
        Path('postcoupling_compute').touch()
        outputs['func_aero'] = np.sum(inputs['f_aero'] + inputs['prestate_aero'] + inputs['x_aero'])


class AeroBuilder(Builder):
    def get_number_of_nodes(self):
        return num_nodes

    def get_ndof(self):
        return 3

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return MeshComp()

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return PreCouplingComp()

    def get_coupling_group_subsystem(self, scenario_name=None):
        return CouplingComp()

    def get_post_coupling_subsystem(self, scenario_name=None):
        return PostCouplingComp()


class Geometry(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_aero_in', shape_by_conn=True)
        self.add_output('x_aero0', shape=3*num_nodes, tags=['mphys_coordinates'])

    def compute(self, inputs, outputs):
        outputs['x_aero0'] = inputs['x_aero_in']


class GeometryBuilder(Builder):
    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return Geometry()

def remove_dir(dir_name):
    shutil.rmtree(dir_name)

def make_dir(dir_name):
    Path(dir_name).mkdir(exist_ok=True)


class TestScenarioAerodynamic(unittest.TestCase):
    # don't want multiple procs to get out of sync since using file creation/removal
    N_PROCS = 1

    def setUp(self):
        self.scenarios = ['cruise', 'maneuver']
        for scenario in self.scenarios:
            print('MAKE_DIR')
            make_dir(scenario)
        self.common = CommonMethods()
        self.prob = om.Problem()
        builder = AeroBuilder()
        builder.initialize(MPI.COMM_WORLD)
        self.prob.model.add_subsystem('mesh', builder.get_mesh_coordinate_subsystem())
        for scenario in self.scenarios:
            self.prob.model.add_subsystem(scenario, ScenarioAerodynamic(aero_builder=builder, run_directory=scenario))
            self.prob.model.connect('mesh.x_aero0', f'{scenario}.x_aero')
        self.prob.setup()

    def tearDown(self):
        for scenario in self.scenarios:
            remove_dir(scenario)

    def testRunModel(self):
        self.common.test_run_model(self)
        for scenario in self.scenarios:
            for expected_file in ['precoupling_compute', 'coupling_compute', 'postcoupling_compute']:
                print('LS', subprocess.check_output(['ls']).decode('utf-8'))
                path =f'{scenario}/{expected_file}'
                print(f' PATH {path}')
                print('LS', subprocess.check_output(['ls',scenario]).decode('utf-8'))
                self.assertTrue(Path(f'{scenario}/{expected_file}').exists())



if __name__ == '__main__':
    unittest.main()
