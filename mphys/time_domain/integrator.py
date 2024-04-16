import openmdao.api as om

from .time_domain_builder import TimeDomainBuilder


class Integrator(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("nsteps", types=int, desc="Number of time steps")
        self.options.declare("dt", desc="Time step size")

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
        self._setup_step_problem()

    def _setup_step_problem(self):
        """
        OpenMDAO problem solved at each time step
        """
        self.add_step_inputs()
        self.add_initial_condition_inputs()
        self.problem = om.Problem()
        self.problem.model = self._get_timestep_group()
        self.problem.setup()

    def add_step_inputs(self):
        for builder in self._get_builder_list():
            for var in builder.get_timestep_input_variables():
                self.add_input(var.name, shape=var.shape, distributed=var.distributed)

    def add_initial_condition_inputs(self):
        for builder in self._get_builder_list():
            for var in builder.get_time_derivative_variables():
                self.add_input(
                    f"{var.name}|0", shape=var.shape, distributed=var.distributed
                )

    def compute(self, inputs, outputs):
        self._set_mphys_inputs(inputs)

        # start from step 1 to leave initial conditions as 0
        for step in range(1, self.options["nsteps"] + 1):
            if self.comm.rank == 0:
                print(f"Integrator step {step}")
            self._set_time_step_information(step)
            self._setup_step_backplanes(step, inputs)
            self.problem.run_model()
            self._store_step_output(step)

    def _set_time_step_information(self, step):
        self.problem["dt"] = self.options["dt"]
        self.problem["step"] = step
        self.problem["time"] = step * self.options["dt"]

    def _set_mphys_inputs(self, inputs):
        for builder in self._get_builder_list():
            for var in builder.get_timestep_input_variables():
                self.problem[f"{var.name}"] = inputs[var.name]

    def _setup_step_backplanes(self, step, inputs):
        for builder in self._get_builder_list():
            for var in builder.get_time_derivative_variables():
                for backplane in range(var.number_of_backplanes):
                    prior_step = step - backplane
                    if prior_step > 0:
                        source = "" if backplane == 0 else f"|t-{backplane}"
                        self.problem[f"{var.name}|t-{backplane+1}"] = self.problem[
                            f"{var.name}{source}"
                        ]
                    else:
                        self.problem[f"{var.name}|t-{backplane+1}"] = inputs[
                            f"{var.name}|0"
                        ]

    def _store_step_output(self, step):
        pass

    def _get_timestep_group(self):
        pass

    def _get_builder_list(self) -> list[TimeDomainBuilder]:
        return []
