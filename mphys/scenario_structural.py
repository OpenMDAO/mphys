from .scenario import Scenario

class ScenarioStructural(Scenario):

    def initialize(self):
        self.options.declare('struct_builder', recordable=False)

    def setup(self):
        struct_builder = self.options['struct_builder']

        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)
        self.mphys_add_subsystem('coupling',struct_builder.get_coupling_group_subsystem())
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)
