from mphys.time_domain.integrator import Integrator

from .timestep_aerostructural import TimeStepAeroStructural


class IntegratorAerostructural(Integrator):
    def initialize(self):
        self.options.declare("aero_builder")
        self.options.declare("struct_builder")
        self.options.declare("ldxfer_builder")
        return super().initialize()

    def _get_timestep_group(self):
        return TimeStepAeroStructural(
            aero_builder=self.options["aero_builder"],
            struct_builder=self.options["struct_builder"],
            ldxfer_builder=self.options["ldxfer_builder"],
            nonlinear_solver=self.options["nonlinear_solver"],
            linear_solver=self.options["linear_solver"],
        )

    def _get_builder_list(self):
        return [
            self.options["aero_builder"],
            self.options["struct_builder"],
            self.options["ldxfer_builder"],
        ]
