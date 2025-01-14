from typing import List

import openmdao.api as om

from mphys import Builder


class Geometry(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('disciplines', default=[])
        self.options.declare('discipline_num_nodes', default=[])

    def setup(self):
        for disp, size in zip(self.options['disciplines'],self.options['discipline_num_nodes']):
            self.add_input(f'x_{disp}0_geometry_input', shape_by_conn=True, tags=['mphys_coordinates'])
            self.add_output(f'x_{disp}0_geometry_output', shape=3*size, tags=['mphys_coordinates'])


    def compute(self, inputs, outputs):
        for disp in self.options['disciplines']:
            outputs[f'x_{disp}0_geometry_output'] = inputs[f'x_{disp}0_geometry_input']


class GeometryBuilder(Builder):
    def __init__(self, disciplines: List[str], builders: List[Builder]):
        self.disciplines = disciplines
        self.builders = builders

    def initialize(self, comm):
        self.num_nodes = []
        for builder in self.builders:
            self.num_nodes.append(builder.get_number_of_nodes())

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return Geometry(disciplines=self.disciplines,
                        discipline_num_nodes=self.num_nodes)
