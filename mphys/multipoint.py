import openmdao.api as om

class Multipoint(om.Group):
    def mphys_add_scenario(self, name, scenario, coupling_nonlinear_solver=None,
                                                 coupling_linear_solver=None):
        scenario = self.add_subsystem(name,scenario)

        if coupling_nonlinear_solver is not None:
            scenario.coupling.nonlinear_solver = coupling_nonlinear_solver

        if coupling_linear_solver is not None:
            scenario.coupling.nonlinear_solver = coupling_linear_solver

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
                src = "%s.x_%s0" % (source, discipline)
                target = "%s.x_%s0" % (scenario, discipline)
                self.connect(src,target)
