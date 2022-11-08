from typing import List
import openmdao.api as om
from mphys import Builder

class Geometry(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('disciplines', default=[])
        self.options.declare('discipline_num_nodes', default=[])

    def setup(self):
        for disp, size in zip(self.options['disciplines'],self.options['discipline_num_nodes']):
            self.add_input(f'x_{disp}_in', shape_by_conn=True)
            self.add_output(f'x_{disp}0', shape=3*size, tags=['mphys_coordinates'])


    def compute(self, inputs, outputs):
        for disp in self.options['disciplines']:
            outputs[f'x_{disp}0'] = inputs[f'x_{disp}_in']


class GeometryBuilder(Builder):
    def __init__(self, disciplines: List[str], builders: List[Builder]):
        self.disciplines = disciplines
        self.builders = builders

    def initialize(self, comm):
        self.num_nodes = []
        for builder in self.builders:
            print('BUILDER', type(builder))
            self.num_nodes.append(builder.get_number_of_nodes())

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return Geometry(disciplines=self.disciplines,
                        discipline_num_nodes=self.num_nodes)
