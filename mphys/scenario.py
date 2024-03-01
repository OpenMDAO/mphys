from functools import wraps

from .mphys_group import MphysGroup
from .utils.directory_utils import cd

# Define decorator functions for methods where run directory must be switched before calling
def switch_run_directory(method):
    @wraps(method)
    def wrapped_method(self, *args, **kwargs):
        with cd(self.options['run_directory']):
            return method(self, *args, **kwargs)

    return wrapped_method

class Scenario(MphysGroup):
    """
    A group to represent a specific analysis condition or point of the MPhys
    multipoint groups.

    To make a Scenario for a particular type of multiphysics problem, subclass
    the Scenario, and implement the `initialize` and `setup` phases of the Group.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._post_subsystems = []

    def initialize(self):
        self.options.declare('run_directory',
                             default='',
                             types=str,
                             desc='Path in which to execute subsystems in this scenario group.' +
                                  ' The default of empty string will not change the directory.')

    def _mphys_scenario_setup(self):
        """
        This function is where specific scenarios populate pre-coupling, coupling,
        and post-coupling subsystems from builders
        """
        pass

    def setup(self):
        """
        The main setup call for all multiphysics scenarios.
        Multiphysics scenarios should implement setup-type operations in _mphys_scenario_setup().
        Adds the builder subsystems, then adds user-defined post subsystems.
        """
        self._mphys_scenario_setup()
        self._add_post_subsystems()

    def mphys_add_post_subsystem(self, name, subsystem,
                                 promotes_inputs=None,
                                 promotes_outputs=None,
                                 promotes=None):
        """
        Add a user-defined subsystem at the end of a Scenario.
        Tag variables with mphys tags to promote or use the optional promotes argument.

        Parameters
        ----------
        name: str
            Name of the subsystem
        subsystem: <System>
            The
        promotes: iter of (str or tuple), optional
            If None, variables will be promoted using mphys_* tags,
            else variables will be promoted by this input
        """

        # we hold onto these until the end of setup() b/c we want the scenario's
        # setup() to add the builder subsystems before adding these
        self._post_subsystems.append((name, subsystem, promotes_inputs, promotes_outputs, promotes))

    @switch_run_directory
    def _solve_nonlinear(self, *args, **kwargs):
        return super()._solve_nonlinear(*args, **kwargs)

    @switch_run_directory
    def _solve_linear(self, *args, **kwargs):
        return super()._solve_linear(*args, **kwargs)

    @switch_run_directory
    def _apply_nonlinear(self, *args, **kwargs):
        return super()._apply_nonlinear(*args, **kwargs)

    @switch_run_directory
    def _apply_linear(self, *args, **kwargs):
        return super()._apply_linear(*args, **kwargs)

    def _mphys_add_pre_coupling_subsystem_from_builder(self, name, builder, scenario_name=None):
        """
        If the builder has a precoupling subsystem, add it to the model.
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

    def _mphys_add_post_coupling_subsystem_from_builder(self, name, builder, scenario_name=None):
        """
        If the builder has a postcoupling subsystem, add it to the model.
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

    def _add_post_subsystems(self):
        for name, subsystem, promotes_inputs, promotes_outputs, promotes in self._post_subsystems:
            if self._no_promotes_specified(promotes_inputs, promotes_outputs, promotes):
                self.mphys_add_subsystem(name, subsystem)
            else:
                self.add_subsystem(name, subsystem,
                                   promotes_inputs=promotes_inputs,
                                   promotes_outputs=promotes_outputs,
                                   promotes=promotes)

    def _no_promotes_specified(self, promotes_inputs, promotes_outputs, promotes):
        return (promotes_inputs is None and
                promotes_outputs is None and
                promotes is None)
