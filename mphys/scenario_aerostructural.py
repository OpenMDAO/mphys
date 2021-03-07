import openmdao.api as om
from .scenario import Scenario
from .coupling_aerostructural import CouplingAeroStructural

class ScenarioAeroStructural(Scenario):

    def initialize(self):
        self.options.declare('aero_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('ldxfer_builder', recordable=False)

    def setup(self):
        aero_builder = self.options['aero_builder']
        struct_builder = self.options['struct_builder']
        ldxfer_builder = self.options['ldxfer_builder']

        self.mphys_add_pre_coupling_subsystem('aero', aero_builder)
        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)
        self.mphys_add_pre_coupling_subsystem('ldxfer', ldxfer_builder)

        coupling_group = CouplingAeroStructural(aero_builder=aero_builder,
                                                struct_builder=struct_builder,
                                                ldxfer_builder=ldxfer_builder)
        self.mphys_add_subsystem('coupling',coupling_group)

        self.mphys_add_post_coupling_subsystem('aero', aero_builder)
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)
        self.mphys_add_post_coupling_subsystem('ldxfer', ldxfer_builder)

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, atol=1e-8, rtol=1e-8, use_aitken=True)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, atol=1e-8, rtol=1e-8, use_aitken=True)
