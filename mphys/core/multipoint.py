import openmdao.api as om


def set_coupling_algorithms_in_scenarios(multipoint_group):
    """
    Set the stored linear and nonlinear solver into the coupling group if the
    scenario exists on this proc.
    Shared method between multipoint and multipoint parallel groups.
    Called during configure().
    """
    for scenario, solvers in multipoint_group.mphys_coupling_solvers:
        if solvers[0] is not None and scenario.comm:
            scenario.coupling.nonlinear_solver = solvers[0]

        if solvers[1] is not None and scenario.comm:
            scenario.coupling.linear_solver = solvers[1]


class Multipoint(om.Group):
    """
    An extension of the standard OpenMDAO group that adds the :func:`~mphys_add_scenario` method.
    For sequential evaluations of the MPhys scenarios.
    """

    def __init__(self, **kwargs):
        self.mphys_coupling_solvers = []
        super().__init__(**kwargs)

    def mphys_add_scenario(self, name, scenario, coupling_nonlinear_solver=None,
                           coupling_linear_solver=None):
        """
        Add an MPhys scenario

        Parameters
        ----------
        name : str
            The name of the scenario
        Scenario: :class:`~mphys.scenario.Scenario`
            The scenario object
        coupling_nonlinear_solver: openmdao.solvers.solver.NonlinearSolver
            The nonlinear solver to assign to the coupling group primal problem
        coupling_linear_solver: openmdao.solvers.solver.LinearSolver
            The linear solver to to assign to the coupling group sensitivity problem
        """
        solver_tuple = (coupling_nonlinear_solver, coupling_linear_solver)
        self.mphys_coupling_solvers.append((scenario, solver_tuple))
        return self.add_subsystem(name, scenario)

    def configure(self):
        return set_coupling_algorithms_in_scenarios(self)


class MultipointParallel(om.ParallelGroup):
    """
    An OpenMDAO parallel group that adds the :func:`~mphys_add_scenario` method.
    For simultaneous evaluations of the MPhys scenarios.
    """

    def __init__(self, **kwargs):
        self.mphys_coupling_solvers = []
        super().__init__(**kwargs)

    def mphys_add_scenario(self, name, scenario, coupling_nonlinear_solver=None,
                           coupling_linear_solver=None):
        """
        Add an MPhys scenario

        Parameters
        ----------
        name : str
            The name of the scenario
        Scenario: :class:`~mphys.scenario.Scenario`
            The scenario object
        coupling_nonlinear_solver: openmdao.solvers.solver.NonlinearSolver
            The nonlinear solver to assign to the coupling group primal problem
        coupling_linear_solver: openmdao.solvers.solver.LinearSolver
            The linear solver to to assign to the coupling group sensitivity problem
        """
        solver_tuple = (coupling_nonlinear_solver, coupling_linear_solver)
        self.mphys_coupling_solvers.append((scenario, solver_tuple))
        return self.add_subsystem(name, scenario)

    def configure(self):
        return set_coupling_algorithms_in_scenarios(self)
