from .scenario import Scenario

class ScenarioStructural(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline structural case
        for structural solvers which can compute their own set of loads.
        The Scenario will add the structural builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        self.options.declare('struct_builder', recordable=False,
                             desc='The Mphys builder for the structural solver')
        self.options.declare('in_MultipointParallel', default=False, types=bool,
                             desc='Set to `True` if adding this scenario inside a MultipointParallel Group.')
        self.options.declare('geometry_builder', default=None, recordable=False,
                             desc='The optional Mphys builder for the geometry')

    def setup(self):
        struct_builder = self.options['struct_builder']
        geometry_builder = self.options['geometry_builder']

        if self.options['in_MultipointParallel']:
            struct_builder.initialize(self.comm)

            if geometry_builder is not None:
                geometry_builder.initialize(self.comm)
                self.add_subsystem('mesh',struct_builder.get_mesh_coordinate_subsystem(self.name))
                self.mphys_add_subsystem('geometry',geometry_builder.get_mesh_coordinate_subsystem(self.name))
                self.connect('mesh.x_struct0','geometry.x_struct_in')
            else:
                self.mphys_add_subsystem('mesh',struct_builder.get_mesh_coordinate_subsystem(self.name))

        self.mphys_add_pre_coupling_subsystem('struct', struct_builder, self.name)
        self.mphys_add_subsystem('coupling',struct_builder.get_coupling_group_subsystem(self.name))
        self.mphys_add_post_coupling_subsystem('struct', struct_builder, self.name)
