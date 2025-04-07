from typing import List

import numpy as np
import openmdao.api as om
from mpi4py import MPI

from mphys import Builder, MPhysGeometry


class GeometryMorph(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "discipline_geometries",
            types=list,
            desc="list of MPhysGeometry classes for the disciplines from the MPhys variable convention",
        )
        self.options.declare(
            "discipline_builders",
            types=list,
            desc="list of Builder classes for the disciplines",
        )

    def setup(self):
        self.add_input("geometry_morph_param")
        self.discipline_geometries: List[MPhysGeometry] = self.options[
            "discipline_geometries"
        ]
        discipline_builders: List[Builder] = self.options["discipline_builders"]

        for geom, builder in zip(self.discipline_geometries, discipline_builders):
            self.add_input(geom.COORDINATES_INPUT, distributed=True, shape_by_conn=True)

            self.add_output(
                geom.COORDINATES_OUTPUT,
                shape=3 * builder.get_number_of_nodes(),
                distributed=True,
                tags=["mphys_coordinates"],
            )

    def compute(self, inputs, outputs):
        for discip in self.discipline_geometries:
            outputs[discip.COORDINATES_OUTPUT] = (
                inputs["geometry_morph_param"] * inputs[discip.COORDINATES_INPUT]
            )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            for discip in self.discipline_geometries:
                if discip.COORDINATES_OUTPUT in d_outputs:
                    if "geometry_morph_param" in d_inputs:
                        d_inputs["geometry_morph_param"] += self.comm.allreduce(
                            np.sum(
                                d_outputs[discip.COORDINATES_OUTPUT]
                                * inputs[discip.COORDINATES_INPUT]
                            ),
                            op=MPI.SUM,
                        )

                    if discip.COORDINATES_INPUT in d_inputs:
                        d_inputs[discip.COORDINATES_INPUT] += (
                            d_outputs[discip.COORDINATES_OUTPUT]
                            * inputs["geometry_morph_param"]
                        )


class GeometryBuilder(Builder):
    def __init__(self, discipline_builders=None, discipline_geometries=None):
        self.discipline_builders = discipline_builders or []
        self.discipline_geometries = discipline_geometries or []

    def add_discipline(self, builder: Builder, geometry):
        self.discipline_geometries.append(geometry)
        self.discipline_builders.append(builder)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return GeometryMorph(
            discipline_builders=self.discipline_builders,
            discipline_geometries=self.discipline_geometries,
        )
