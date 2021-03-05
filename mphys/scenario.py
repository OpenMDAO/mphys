from .mphys_group import MphysGroup

class Scenario(MphysGroup):
    def mphys_add_pre_coupling_subsystem(self, name, builder):
        subsystem, _ = builder.get_scenario_subsystems()
        if subsystem is not None:
            self.mphys_add_subsystem(name+'_pre', subsystem)

    def mphys_add_post_coupling_subsystem(self, name, builder):
        _, subsystem = builder.get_scenario_subsystems()
        if subsystem is not None:
            self.mphys_add_subsystem(name+'_post', subsystem)
