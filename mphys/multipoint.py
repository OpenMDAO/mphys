import openmdao.api as om

class MultipointBase:
    def __init__(self):
        self.mphys_coupling_solvers = []

    def mphys_add_scenario(self, name, scenario, coupling_nonlinear_solver=None,
                                                 coupling_linear_solver=None):
        solver_tuple = (coupling_nonlinear_solver, coupling_linear_solver)
        self.mphys_coupling_solvers.append((scenario,solver_tuple))
        return self.add_subsystem(name,scenario)

    def configure(self):
        self._mphys_set_coupling_algorithms_in_scenarios()

    def _mphys_set_coupling_algorithms_in_scenarios(self):
        for scenario, solvers in self.mphys_coupling_solvers:
            if solvers[0] is not None:
                scenario.coupling.nonlinear_solver = solvers[0]

            if solvers[1] is not None:
                scenario.coupling.linear_solver = solvers[1]

class Multipoint(om.Group,MultipointBase):
    def __init__(self, **kwargs):
        MultipointBase.__init__(self)
        om.Group.__init__(self, **kwargs)

    def mphys_connect_scenario_coordinate_source(self, source, scenarios, disciplines):
        """

        Parameters
        ----------
        disciplines : str or list[str]
            The extension of the after the underscore in x_ for
        source: openmdao.api.Group or openmdao.api.Component

        scenario_list : str or list[str]

        """
        scenarios_list = scenarios if type(scenarios) == list else [scenarios]
        disciplines_list = disciplines if type(disciplines) == list else [disciplines]

        for scenario in scenarios_list:
            for discipline in disciplines_list:
                src = f'{source}.x_{discipline}0'
                target = f'{scenario}.x_{discipline}0'
                self.connect(src,target)

    def configure(self):
        return MultipointBase.configure(self)

class MultipointParallel(om.ParallelGroup, MultipointBase):
    def __init__(self, **kwargs):
        MultipointBase.__init__(self)
        om.ParallelGroup.__init__(self, **kwargs)

    def configure(self):
        return MultipointBase.configure(self)
