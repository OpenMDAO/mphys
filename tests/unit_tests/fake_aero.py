import numpy as np
import openmdao.api as om
from mphys import Builder, MPhysVariables

aero_num_nodes = 3


class AeroMeshComp(om.IndepVarComp):
    def setup(self):
        self.add_output(
            MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
            val=np.ones(aero_num_nodes * 3),
            tags=["mphys_coordinates"],
        )


class AeroPreCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.x_aero_name = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL

        self.add_input(self.x_aero_name, shape_by_conn=True, tags=["mphys_coordinates"])
        self.add_output("prestate_aero", tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        outputs["prestate_aero"] = np.sum(inputs[self.x_aero_name])


class AeroCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.coords_name = MPhysVariables.Aerodynamics.Surface.COORDINATES
        self.loads_name = MPhysVariables.Aerodynamics.Surface.LOADS

        self.add_input(self.coords_name, shape_by_conn=True, tags=["mphys_coordinates"])
        self.add_input("prestate_aero", tags=["mphys_coupling"])
        self.add_output(self.loads_name, shape=aero_num_nodes * 3, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        outputs[self.loads_name] = inputs[self.coords_name] + inputs["prestate_aero"]


class AeroPostCouplingComp(om.ExplicitComponent):
    def setup(self):
        self.coords_name = MPhysVariables.Aerodynamics.Surface.COORDINATES
        self.loads_name = MPhysVariables.Aerodynamics.Surface.LOADS

        self.add_input("prestate_aero", tags=["mphys_coupling"])
        self.add_input(self.coords_name, shape_by_conn=True, tags=["mphys_coordinates"])
        self.add_input(self.loads_name, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("func_aero", val=1.0, tags=["mphys_result"])

    def compute(self, inputs, outputs):
        outputs["func_aero"] = np.sum(inputs[self.loads_name] + inputs["prestate_aero"] + inputs[self.coords_name])


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


if __name__ == "__main__":
    aero_builder = AeroBuilder()
    print(aero_builder.get_number_of_nodes())
