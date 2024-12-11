import numpy as np
import openmdao.api as om

from mphys import Builder, MPhysVariables

struct_num_nodes = 3


class StructMeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output(
            MPhysVariables.Structures.Mesh.COORDINATES, val=np.ones(struct_num_nodes * 3), tags=["mphys_coordinates"]
        )


class StructPreCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.coords_name = MPhysVariables.Structures.COORDINATES
        self.add_input(self.coords_name, shape_by_conn=True, tags=["mphys_coordinates"])
        self.add_output("prestate_struct", tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        outputs["prestate_struct"] = np.sum(inputs[self.coords_name])


class StructCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.coords_name = MPhysVariables.Structures.COORDINATES
        self.aero_loads_name = MPhysVariables.Structures.Loads.AERODYNAMIC
        self.disps_name = MPhysVariables.Structures.DISPLACEMENTS

        self.add_input(self.coords_name, shape_by_conn=True, tags=["mphys_coordinates"])
        self.add_input("prestate_struct", tags=["mphys_coupling"])
        self.add_input(self.aero_loads_name, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output(self.disps_name, shape=struct_num_nodes * 3, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        outputs[self.disps_name] = inputs[self.coords_name] + inputs["prestate_struct"]


class StructPostCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.coords_name = MPhysVariables.Structures.COORDINATES
        self.disps_name = MPhysVariables.Structures.DISPLACEMENTS

        self.add_input("prestate_struct", tags=["mphys_coupling"])
        self.add_input(self.coords_name, shape_by_conn=True, tags=["mphys_coordinates"])
        self.add_input(self.disps_name, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("func_struct", val=1.0, tags=["mphys_result"])

    def compute(self, inputs, outputs):
        outputs["func_struct"] = np.sum(inputs[self.disps_name] + inputs["prestate_struct"] + inputs[self.coords_name])


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
