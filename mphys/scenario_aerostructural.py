from .scenario import Scenario
from .coupling_aerostructural import CouplingAeroStructural

class ScenarioAeroStructural(Scenario):

    def initialize(self):
        self.options.declare('aero_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('xfer_builder', recordable=False)

    def setup(self):
        aero_builder = self.options['aero_builder']
        struct_builder = self.options['struct_builder']
        xfer_builder = self.options['xfer_builder']

        self.mphys_add_pre_coupling_subsystem('aero', aero_builder)
        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)
        self.mphys_add_pre_coupling_subsystem('xfer', xfer_builder)

        coupling_group = CouplingAeroStructural(aero_builder=aero_builder,
                                                struct_builder=struct_builder,
                                                xfer_builder=xfer_builder)
        self.mphys_add_subsystem('coupling',coupling_group)

        self.mphys_add_post_coupling_subsystem('aero', aero_builder)
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)
        self.mphys_add_post_coupling_subsystem('xfer', xfer_builder)
