import openmdao.api as om
from mphys import Builder

class DispXferComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('aero_num_nodes', default=3)

    def setup(self):
        aero_num_nodes = self.options['aero_num_nodes']
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('u_aero', shape=aero_num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['u_aero'] = inputs['u_struct']


class LoadXferComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('struct_num_nodes', default=3)

    def setup(self):
        struct_num_nodes = self.options['struct_num_nodes']
        self.add_input('x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('x_aero0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input('u_struct', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('f_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output('f_struct', shape=struct_num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['f_struct'] = inputs['f_aero']


class LDXferBuilder(Builder):
    def __init__(self, aero_builder: Builder, struct_builder: Builder):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder

    def get_coupling_group_subsystem(self, scenario_name=None):
        aero_num_nodes = self.aero_builder.get_number_of_nodes()
        struct_num_nodes = self.struct_builder.get_number_of_nodes()
        return (DispXferComp(aero_num_nodes=aero_num_nodes),
                LoadXferComp(struct_num_nodes=struct_num_nodes))
