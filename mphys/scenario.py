from .mphys_group import MphysGroup
from .utils.directory_utils import cd

class Scenario(MphysGroup):
    """
    A group to represent a specific analysis condition or point of the Mphys
    multipoint groups.

    To make a Scenario for a particular type of multiphysics problem, subclass
    the Scenario, and implement the `initialize` and `setup` phases of the Group.
    """
    def initialize(self):
        self.options.declare('run_directory', default='', types=str)

    def mphys_add_pre_coupling_subsystem(self, name, builder, scenario_name=None):
        """
        If the builder has a precoupling subsystem add it to the model.
        It is expected that is method is called during this scenario's`setup` phase.

        Parameters
        ----------
        name: str
            Name of the discipline
        builder: :class:`~mphys.builder.Builder`
            The discipline's builder object
        scenario_name: str or None
            Name of the scenario being setup (optional)
        """
        subsystem = builder.get_pre_coupling_subsystem(scenario_name)
        if subsystem is not None:
            self.mphys_add_subsystem(name+'_pre', subsystem)

    def mphys_add_post_coupling_subsystem(self, name, builder, scenario_name=None):
        """
        If the builder has a postcoupling subsystem add it to the model.
        It is expected that is method is called during this scenario's`setup` phase.

        Parameters
        ----------
        name: str
            Name of the discipline
        builder: :class:`~mphys.builder.Builder`
            The discipline's builder object
        scenario_name: str or None
            Name of the scenario being setup (optional)
        """
        subsystem = builder.get_post_coupling_subsystem(scenario_name)
        if subsystem is not None:
            self.mphys_add_subsystem(name+'_post', subsystem)

    def _solve_nonlinear(self):
        with cd(self.options['run_directory']):
            return super()._solve_nonlinear()

    def _solve_linear(self, mode, rel_systems, scope_out=..., scope_in=...):
        with cd(self.options['run_directory']):
            return super()._solve_linear(mode, rel_systems, scope_out, scope_in)

    def _apply_nonlinear(self):
        with cd(self.options['run_directory']):
            return super()._apply_nonlinear()

    def _apply_linear(self, jac, rel_systems, mode, scope_out=None, scope_in=None):
        with cd(self.options['run_directory']):
            return super()._apply_linear(jac, rel_systems, mode, scope_out, scope_in)
