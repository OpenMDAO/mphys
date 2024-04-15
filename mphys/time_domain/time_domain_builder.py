from mphys.builder import Builder
from .time_domain_variables import TimeDerivativeVariable, TimeDomainInput


class TimeDomainBuilder(Builder):
    """
    MPHYS time domain builder base class. Template for developers to create their builders.
    """

    def get_pre_integration_subsystem(self, scenario_name=None):
        return None

    def get_post_integration_subsystem(self, scenario_name=None):
        return None

    def get_time_derivative_variables(
        self, scenario_name=None
    ) -> list[TimeDerivativeVariable]:
        """
        The variables associated with this discipline that need backplanes of
        data for time derivatives
        """
        return []

    def get_timestep_input_variables(self, scenario_name=None) -> list[TimeDomainInput]:
        """
        The variables associated with this discipline that are inputs to the time step subsystems
        """
        return []
