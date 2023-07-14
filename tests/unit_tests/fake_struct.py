import numpy as np
import openmdao.api as om

from mphys import Builder

struct_num_nodes = 3

class StructMeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output('x_struct0', val=np.ones(struct_num_nodes*3), tags=['mphys_coordinates'])


class StructPreCouplingComp(om.ExplicitComponent):
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
        self.add_output('u_struct', shape=struct_num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['u_struct'] = inputs['x_struct0'] + inputs['prestate_struct']


class StructPostCouplingComp(om.ExplicitComponent):
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
        return struct_num_nodes

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
