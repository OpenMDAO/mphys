from .scenario import Scenario

class ScenarioStructuralV1(Scenario):

    def initialize(self):
        self.options.declare('struct_builder', recordable=False)

    def setup(self):
        struct_builder = self.options['struct_builder']

        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)

        # the "coupling" group for struct_only would just have the struct subsystem so add it directly here.
        self.mphys_add_subsystem('coupling',struct_builder.get_coupling_group_subsystem())
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)





# a class that works for both Multipoint and MultipointParallel
class ScenarioStructural(Scenario):

    def initialize(self):
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('in_MultipointParallel', default=False)

    def setup(self):
        struct_builder = self.options['struct_builder']

        if self.options['in_MultipointParallel']:
            struct_builder.initialize(self.comm)
            self.mphys_add_subsystem('mesh',struct_builder.get_mesh_coordinate_subsystem())

        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)
        self.mphys_add_subsystem('coupling',struct_builder.get_coupling_group_subsystem())
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)






# UNTESTED: to show in_MultipointParallel option isn't necessary and add geometry
class ScenarioStructuralParallelGeometry(Scenario):

    def initialize(self):
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('geometry_builder', recordable=False)

    def setup(self):
        struct_builder = self.options['struct_builder']
        geometry_builder = self.options['geometry_builder']

        struct_builder.initialize(self.comm)
        geometry_builder.initialize(self.comm)

        # don't use mphys_add_subsystem for mesh so that the geometry's coordinate output
        # are promoted, not the mesh
        self.add_subsystem('mesh',struct_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_subsystem('geometry',geometry_builder.get_mesh_coordinate_subsystem())
        self.connect('mesh.x_struct0','geometry.x_struct_in')

        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)
        self.mphys_add_subsystem('coupling',struct_builder.get_coupling_group_subsystem())
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)
