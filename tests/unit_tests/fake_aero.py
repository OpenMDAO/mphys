import numpy as np
import openmdao.api as om
from mphys import Builder

aero_num_nodes = 3

class AeroMeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output('x_aero0', val=np.ones(aero_num_nodes*3), tags=['mphys_coordinates'])


class AeroPreCouplingComp(om.IndepVarComp):
    def setup(self):
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output('prestate_aero', tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['prestate_aero'] = np.sum(inputs['x_aero'])


class AeroCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('prestate_aero', tags=['mphys_coupling'])
        self.add_output('f_aero', shape=aero_num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['f_aero'] = inputs['x_aero'] + inputs['prestate_aero']


class AeroPostCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('prestate_aero', tags=['mphys_coupling'])
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('f_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('func_aero', val=1.0, tags=['mphys_result'])

    def compute(self, inputs, outputs):
        outputs['func_aero'] = np.sum(inputs['f_aero'] + inputs['prestate_aero'] + inputs['x_aero'])

class AeroBuilder(Builder):

    def get_number_of_nodes(self):
        return aero_num_nodes

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


if __name__ == '__main__':
    aero_builder = AeroBuilder()
    print(aero_builder.get_number_of_nodes())
