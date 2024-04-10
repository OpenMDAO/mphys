import openmdao.api as om
from mphys.mphys_group import MphysGroup

from .time_domain_builder import TimeDomainBuilder
from .time_domain_variables import TimeDomainInput


class TimeStep(MphysGroup):
    def initialize(self):
        self.options.declare("user_input_variables", default=[])
        self.options.declare("scenario_name", default=None)
        self.options.declare(
            "nonlinear_solver",
            default=None,
            desc="Nonlinear solver to use in the time step coupling group",
        )
        self.options.declare(
            "linear_solver",
            default=None,
            desc="Linear solver to use in the time step coupling group",
        )

    def setup(self):
        self._mphys_timestep_setup()

    def configure(self):
        if self.options["linear_solver"] is not None:
            self.coupling.linear_solver = self.options["linear_solver"]
        if self.options["nonlinear_solver"] is not None:
            self.coupling.nonlinear_solver = self.options["nonlinear_solver"]

        return super().configure()

    def _mphys_timestep_setup(self):
        pass

    def _mphys_add_pre_coupling_subsystem_from_builder(
        self, name, builder: TimeDomainBuilder
    ):
        subsystem = builder.get_pre_coupling_subsystem(self.options["scenario_name"])
        if subsystem is not None:
            self.mphys_add_subsystem(name + "_pre", subsystem)

    def _mphys_add_post_coupling_subsystem_from_builder(
        self, name, builder: TimeDomainBuilder
    ):
        subsystem = builder.get_post_coupling_subsystem(self.options["scenario_name"])
        if subsystem is not None:
            self.mphys_add_subsystem(name + "_post", subsystem)

    def _add_ivc_with_mphys_inputs(
        self,
        builders: list[TimeDomainBuilder],
        user_inputs: list[TimeDomainInput],
    ):
        ivc = om.IndepVarComp()
        for builder in builders:
            for var in builder.get_timestep_input_variables(
                self.options["scenario_name"]
            ):
                ivc.add_output(var.name, shape=var.shape)
                print(f"DEBUG: added {var.name}, {var.shape}")

        for var in user_inputs:
            ivc.add_output(var.name, shape=var.shape)
        self.add_subsystem("timestep_inputs", ivc, promotes=["*"])

    def _add_ivc_with_time_information(self):
        ivc = om.IndepVarComp()
        ivc.add_output("step")
        ivc.add_output("dt")
        ivc.add_output("time")
        self.add_subsystem("time_information", ivc, promotes=["*"])

    def _add_ivc_with_state_backplanes(self, builders: list[TimeDomainBuilder]):
        ivc = om.IndepVarComp()
        for builder in builders:
            for var in builder.get_time_derivative_variables(
                self.options["scenario_name"]
            ):
                for n in range(var.number_of_backplanes):
                    name = f"{var.name}|t-{n+1}"
                    ivc.add_output(name, shape=var.shape)
                    print(f"DEBUG: added {name}")
        self.add_subsystem("time_derivative_inputs", ivc, promotes=["*"])
