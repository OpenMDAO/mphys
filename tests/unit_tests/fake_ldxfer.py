import openmdao.api as om
from mphys import Builder, MPhysVariables

class DispXferComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('aero_num_nodes', default=3)

    def setup(self):
        self.x_aero0_name = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
        self.u_aero_name = MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS

        self.x_struct0_name = MPhysVariables.Structures.COORDINATES
        self.u_struct_name = MPhysVariables.Structures.DISPLACEMENTS

        aero_num_nodes = self.options['aero_num_nodes']
        self.add_input(self.x_struct0_name, shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input(self.x_aero0_name, shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input(self.u_struct_name, shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output(self.u_aero_name, shape=aero_num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs[self.u_aero_name] = inputs[self.u_struct_name]


class LoadXferComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('struct_num_nodes', default=3)

    def setup(self):
        self.x_aero0_name = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
        self.f_aero_name = MPhysVariables.Aerodynamics.Surface.LOADS

        self.x_struct0_name = MPhysVariables.Structures.COORDINATES
        self.u_struct_name = MPhysVariables.Structures.DISPLACEMENTS
        self.f_struct_name = MPhysVariables.Structures.LOADS_FROM_AERODYNAMICS

        struct_num_nodes = self.options['struct_num_nodes']
        self.add_input(self.x_struct0_name, shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input(self.x_aero0_name, shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_input(self.u_struct_name, shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input(self.f_aero_name, shape_by_conn=True, tags=['mphys_coupling'])
        self.add_output(self.f_struct_name, shape=struct_num_nodes*3, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs[self.f_struct_name] = inputs[self.f_aero_name]


class LDXferBuilder(Builder):
    def __init__(self, aero_builder: Builder, struct_builder: Builder):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder

    def get_coupling_group_subsystem(self, scenario_name=None):
        aero_num_nodes = self.aero_builder.get_number_of_nodes()
        struct_num_nodes = self.struct_builder.get_number_of_nodes()
        return (DispXferComp(aero_num_nodes=aero_num_nodes),
                LoadXferComp(struct_num_nodes=struct_num_nodes))
