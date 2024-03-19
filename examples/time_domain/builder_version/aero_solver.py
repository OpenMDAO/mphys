#!/usr/bin/env python
import numpy as np
import openmdao.api as om
from mphys.time_domain.time_domain_builder import TimeDomainBuilder
from mphys.time_domain.time_domain_variables import TimeDomainInput


class HarmonicForcer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_nodes")
        self.options.declare("c1", default=1e-3)

    def setup(self):
        self.nnodes = self.options["number_of_nodes"]
        self.c1 = self.options["c1"]

        self.add_input(
            "amplitude_aero",
            shape_by_conn=True,
            tags=["mphys_input"],
            desc="amplitude_aero",
        )
        self.add_input(
            "freq_aero", shape_by_conn=True, tags=["mphys_input"], desc="frequency"
        )
        self.add_input("time", tags=["mphys_input"], desc="current time")
        self.add_input("x_aero", shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("f_aero", shape=self.nnodes * 3, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        amp = inputs["amplitude_aero"]
        freq = inputs["freq_aero"]
        time = inputs["time"]

        outputs["f_aero"] = (
            amp * np.sin(freq * time) * np.ones(self.nnodes * 3)
            - self.c1 * inputs["x_aero"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        amp = inputs["amplitude_aero"]
        freq = inputs["freq_aero"]
        time = inputs["time"]

        if mode == "fwd":
            if "f_aero" in d_outputs:
                if "amplitude_aero" in d_inputs:
                    d_outputs["f_aero"] += (
                        np.sin(freq * time)
                        * np.ones(self.nnodes * 3)
                        * d_inputs["amplitude_aero"]
                    )
                if "freq_aero" in d_inputs:
                    d_outputs["f_aero"] += (
                        time
                        * np.cos(freq * time)
                        * np.ones(self.nnodes * 3)
                        * d_inputs["freq_aero"]
                    )
                if "time" in d_inputs:
                    d_outputs["f_aero"] += (
                        freq
                        * np.cos(freq * time)
                        * np.ones(self.nnodes * 3)
                        * d_inputs["time"]
                    )
                if "x_aero" in d_inputs:
                    d_outputs["f_aero"] -= self.c1 * d_inputs["x_aero"]

        if mode == "rev":
            if "f_aero" in d_outputs:
                if "amplitude_aero" in d_inputs:
                    d_inputs["amplitude_aero"] += np.sin(freq * time) * np.sum(
                        d_outputs["f_aero"]
                    )
                if "freq_aero" in d_inputs:
                    d_inputs["freq_aero"] += (
                        time * np.cos(freq * time) * np.sum(d_outputs["f_aero"])
                    )
                if "time" in d_inputs:
                    d_inputs["time"] += (
                        freq * np.cos(freq * time) * np.sum(d_outputs["f_aero"])
                    )
                if "x_aero" in d_inputs:
                    d_inputs["x_aero"] -= self.c1 * d_outputs["f_aero"]


class FakeAeroBuilder(TimeDomainBuilder):
    def __init__(self, root_name, nmodes, dt):
        self.nmodes = nmodes
        self.dt = dt
        self._read_number_of_nodes(root_name)

    def _read_number_of_nodes(self, root_name):
        filename = f"{root_name}_mode1.dat"
        fh = open(filename)
        while True:
            line = fh.readline()
            if "zone" in line.lower():
                self.nnodes = int(line.split("=")[2].split(",")[0])
                return

    def get_number_of_nodes(self):
        return self.nnodes

    def get_ndof(self):
        return 1

    def get_coupling_group_subsystem(self, scenario_name=None):
        return HarmonicForcer(number_of_nodes=self.nnodes)

    def get_timestep_input_variables(self, scenario_name=None) -> list[TimeDomainInput]:
        return [
            TimeDomainInput("amplitude_aero", (1)),
            TimeDomainInput("freq_aero", (1)),
            TimeDomainInput("x_aero0", (self.nnodes * 3), True),
        ]
