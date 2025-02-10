import numpy as np
import openmdao.api as om
from mpi4py import MPI

from mphys import Builder, MPhysVariables


# EC which morphs the geometry
class GeometryMorph(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("names")
        self.options.declare("n_nodes")

    def setup(self):
        self.add_input("geometry_morph_param")

        self.x_names = {}
        for name, n_nodes in zip(self.options["names"], self.options["n_nodes"]):
            if name=="aero":
                self.x_names[name] = {"input": MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT,
                                      "output": MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT}
            elif name=="struct":
                self.x_names[name] = {"input": MPhysVariables.Structures.Geometry.COORDINATES_INPUT,
                                      "output": MPhysVariables.Structures.Geometry.COORDINATES_OUTPUT}
            self.add_input(
                self.x_names[name]["input"],
                distributed=True,
                shape_by_conn=True,
                tags=["mphys_coordinates"],
            )
            self.add_output(
                self.x_names[name]["output"],
                shape=n_nodes * 3,
                distributed=True,
                tags=["mphys_coordinates"],
            )

    def compute(self, inputs, outputs):
        for name in self.options["names"]:
            outputs[self.x_names[name]["output"]] = (
                inputs["geometry_morph_param"] * inputs[self.x_names[name]["input"]]
            )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            for name in self.options["names"]:
                if self.x_names[name]["output"] in d_outputs:
                    if "geometry_morph_param" in d_inputs:
                        d_inputs["geometry_morph_param"] += self.comm.allreduce(
                            np.sum(d_outputs[self.x_names[name]["output"]] * inputs[self.x_names[name]["input"]]),
                            op=MPI.SUM,
                        )

                    if self.x_names[name]["input"] in d_inputs:
                        d_inputs[self.x_names[name]["input"]] += (
                            d_outputs[self.x_names[name]["output"]] * inputs["geometry_morph_param"]
                        )


# Builder
class GeometryBuilder(Builder):
    def __init__(self, builders):
        self.builders = builders
        self.names = None

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        if self.names is None:
            self.names = []
            self.n_nodes = []
            for name, builder in self.builders.items():
                self.names.append(name)
                self.n_nodes.append(builder.get_number_of_nodes())
        return GeometryMorph(names=self.names, n_nodes=self.n_nodes)
