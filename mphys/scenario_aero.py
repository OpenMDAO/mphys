from .scenario import Scenario

class ScenarioAero(Scenario):
    def initialize(self):
        self.options.declare('aero_builder', recordable=False)

    def setup(self):
        aero_builder = self.options['aero_builder']

        self.mphys_add_pre_coupling_subsystem('aero', aero_builder)
        self.mphys_add_subsystem('coupling',aero_builder.get_coupling_group_subsystem())
        self.mphys_add_post_coupling_subsystem('aero', aero_builder)
