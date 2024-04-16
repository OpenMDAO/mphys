#!/usr/bin/env python
import numpy as np
import openmdao.api as om
from mphys.time_domain.time_domain_builder import TimeDomainBuilder
from mphys.time_domain.time_domain_variables import (
    TimeDerivativeVariable,
    TimeDomainInput,
)


class ModalStep(om.ImplicitComponent):
    """
    Modal structural BDF2 integrator
    Solves one time step of:
      M ddot(z) + C dot(z) + K z - f = 0

    This a simpler model to work on time integration in mphys
    It could be an explicit component but it written as an implicit
    since the FEM solvers will be implicit
    """

    def initialize(self):
        self.options.declare("nmodes", default=1)
        self.options.declare("dt", default=0.0)

    def setup(self):
        # BDF coefficients - 1st and 2nd derivatives
        dt = self.options["dt"]
        self.alpha = np.zeros(3)
        self.alpha[0] = 3.0 / 2.0 / dt
        self.alpha[1] = -2.0 / dt
        self.alpha[2] = 1.0 / 2.0 / dt

        self.beta = np.zeros(5)
        for i in range(3):
            for j in range(3):
                self.beta[i + j] += self.alpha[i] * self.alpha[j]

        # OM setup
        nmodes = self.options["nmodes"]
        self.add_input(
            "m",
            shape=nmodes,
            val=np.ones(nmodes),
            tags=["mphys_input"],
            desc="modal mass",
        )
        self.add_input(
            "c",
            shape=nmodes,
            val=np.ones(nmodes),
            tags=["mphys_input"],
            desc="modal damping",
        )
        self.add_input(
            "k",
            shape=nmodes,
            val=np.ones(nmodes),
            tags=["mphys_input"],
            desc="modal stiffness",
        )

        self.add_input(
            "f_struct", shape_by_conn=True, tags=["mphys_coupling"], desc="modal force"
        )

        self.add_input(
            "u_struct|t-4",
            shape_by_conn=True,
            tags=["mphys_time_derivative"],
            desc="u_struct at step n-4",
        )
        self.add_input(
            "u_struct|t-3",
            shape_by_conn=True,
            tags=["mphys_time_derivative"],
            desc="u_struct at step n-3",
        )
        self.add_input(
            "u_struct|t-2",
            shape_by_conn=True,
            tags=["mphys_time_derivative"],
            desc="u_struct at step n-2",
        )
        self.add_input(
            "u_struct|t-1",
            shape_by_conn=True,
            tags=["mphys_time_derivative"],
            desc="u_struct at step n-1",
        )

        self.add_output(
            "u_struct",
            shape=nmodes,
            val=np.zeros(nmodes),
            tags=["mphys_coupling"],
            desc="current displacement (step n)",
        )

    def _get_accel_and_vel(self, inputs, outputs):
        accel = (
            self.beta[0] * outputs["u_struct"]
            + self.beta[1] * inputs["u_struct|t-1"]
            + self.beta[2] * inputs["u_struct|t-2"]
            + self.beta[3] * inputs["u_struct|t-3"]
            + self.beta[4] * inputs["u_struct|t-4"]
        )

        vel = (
            self.alpha[0] * outputs["u_struct"]
            + self.alpha[1] * inputs["u_struct|t-1"]
            + self.alpha[2] * inputs["u_struct|t-2"]
        )
        return accel, vel

    def apply_nonlinear(self, inputs, outputs, residuals):
        accel, vel = self._get_accel_and_vel(inputs, outputs)

        residuals["u_struct"] = (
            inputs["m"] * accel
            + inputs["c"] * vel
            + inputs["k"] * outputs["u_struct"]
            - inputs["f_struct"]
        )

    def solve_nonlinear(self, inputs, outputs):
        m = inputs["m"]
        c = inputs["c"]
        k = inputs["k"]

        outputs["u_struct"] = (
            inputs["f_struct"]
            - self.beta[1] * m * inputs["u_struct|t-1"]
            - self.beta[2] * m * inputs["u_struct|t-2"]
            - self.beta[3] * m * inputs["u_struct|t-3"]
            - self.beta[4] * m * inputs["u_struct|t-4"]
            - self.alpha[1] * c * inputs["u_struct|t-1"]
            - self.alpha[2] * c * inputs["u_struct|t-2"]
        ) / (self.beta[0] * m + self.alpha[0] * c + k)

    def linearize(self, inputs, outputs, jacobian):
        self.m = inputs["m"]
        self.c = inputs["c"]
        self.k = inputs["k"]

    def solve_linear(self, d_outputs, d_residuals, mode):
        m = self.m
        c = self.c
        k = self.k

        if mode == "fwd":
            d_outputs["u_struct"] = d_residuals["u_struct"] / (
                self.beta[0] * m + self.alpha[0] * c + k
            )

        if mode == "rev":
            d_residuals["u_struct"] = d_outputs["u_struct"] / (
                self.beta[0] * m + self.alpha[0] * c + k
            )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        accel, vel = self._get_accel_and_vel(inputs, outputs)
        m = inputs["m"]
        c = inputs["c"]
        k = inputs["k"]

        if mode == "fwd":
            if "u_struct" in d_residuals:
                if "u_struct" in d_outputs:
                    d_residuals["u_struct"] += (
                        self.beta[0] * m + self.alpha[0] * c + k
                    ) * d_outputs["u_struct"]
                if "m" in d_inputs:
                    d_residuals["u_struct"] += accel * d_inputs["m"]
                if "c" in d_inputs:
                    d_residuals["u_struct"] += vel * d_inputs["c"]
                if "k" in d_inputs:
                    d_residuals["u_struct"] += outputs["u_struct"] * d_inputs["k"]
                if "f" in d_inputs:
                    d_residuals["u_struct"] -= d_inputs["f"]
                if "u_struct|t-4" in d_inputs:
                    d_residuals["u_struct"] += (
                        self.beta[4] * m * d_inputs["u_struct|t-4"]
                    )
                if "u_struct|t-3" in d_inputs:
                    d_residuals["u_struct"] += (
                        self.beta[3] * m * d_inputs["u_struct|t-3"]
                    )
                if "u_struct|t-2" in d_inputs:
                    d_residuals["u_struct"] += (
                        self.beta[2] * m * d_inputs["u_struct|t-2"]
                        + self.alpha[2] * c * d_inputs["u_struct|t-2"]
                    )
                if "u_struct|t-1" in d_inputs:
                    d_residuals["u_struct"] += (
                        self.beta[1] * m * d_inputs["u_struct|t-1"]
                        + self.alpha[1] * c * d_inputs["u_struct|t-1"]
                    )
        if mode == "rev":
            if "u_struct" in d_residuals:
                if "u_struct" in d_outputs:
                    d_outputs["u_struct"] += (
                        self.beta[0] * m + self.alpha[0] * c + k
                    ) * d_residuals["u_struct"]
                if "m" in d_inputs:
                    d_inputs["m"] += accel * d_residuals["u_struct"]
                if "c" in d_inputs:
                    d_inputs["c"] += vel * d_residuals["u_struct"]
                if "k" in d_inputs:
                    d_inputs["k"] += outputs["u_struct"] * d_residuals["u_struct"]
                if "f" in d_inputs:
                    d_inputs["f"] -= d_residuals["u_struct"]
                if "u_struct|t-4" in d_inputs:
                    d_inputs["u_struct|t-4"] += (
                        self.beta[4] * m * d_residuals["u_struct"]
                    )
                if "u_struct|t-3" in d_inputs:
                    d_inputs["u_struct|t-3"] += (
                        self.beta[3] * m * d_residuals["u_struct"]
                    )
                if "u_struct|t-2" in d_inputs:
                    d_inputs["u_struct|t-2"] += (
                        self.beta[2] * m * d_residuals["u_struct"]
                        + self.alpha[2] * c * d_residuals["u_struct"]
                    )
                if "u_struct|t-1" in d_inputs:
                    d_inputs["u_struct|t-1"] += (
                        self.beta[1] * m * d_residuals["u_struct"]
                        + self.alpha[1] * c * d_residuals["u_struct"]
                    )


class ModalStructBuilder(TimeDomainBuilder):
    def __init__(self, nmodes, dt):
        self.nmodes = nmodes
        self.dt = dt

    def get_number_of_nodes(self):
        return self.nmodes

    def get_ndof(self):
        return 1

    def get_coupling_group_subsystem(self, scenario_name=None):
        return ModalStep(nmodes=self.nmodes, dt=self.dt)

    def get_time_derivative_variables(
        self, scenario_name=None
    ) -> list[TimeDerivativeVariable]:
        return [TimeDerivativeVariable("u_struct", 4, (self.nmodes))]

    def get_timestep_input_variables(self, scenario_name=None) -> list[TimeDomainInput]:
        return [
            TimeDomainInput("m", (self.nmodes)),
            TimeDomainInput("c", (self.nmodes)),
            TimeDomainInput("k", (self.nmodes)),
        ]
