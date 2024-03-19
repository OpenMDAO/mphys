#!/usr/bin/env python
import numpy as np
import openmdao.api as om
from mphys.time_domain.time_domain_builder import TimeDomainBuilder


class ModalForces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("nmodes", default=1)
        self.options.declare("mdisp")

    def setup(self):
        self.mdisp = self.options["mdisp"]
        self.nmodes = self.mdisp.shape[0]
        self.nnodes = self.mdisp.shape[1]

        self.add_input(
            "f_aero", shape_by_conn=True, tags=["mphys_coupling"], desc="nodal force"
        )
        self.add_output(
            "f_struct", shape=self.nmodes, tags=["mphys_coupling"], desc="modal force"
        )

    def compute(self, inputs, outputs):
        outputs["f_struct"] = 0.0
        for imode in range(self.nmodes):
            for inode in range(self.nnodes):
                for k in range(3):
                    outputs["f_struct"][imode] += (
                        self.mdisp[imode, inode, k] * inputs["f_aero"][3 * inode + k]
                    )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            if "f_struct" in d_outputs:
                if "f_aero" in d_inputs:
                    for imode in range(self.nmodes):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_outputs["f_struct"][imode] += (
                                    self.mdisp[imode, inode, k]
                                    * d_inputs["f_aero"][3 * inode + k]
                                )
        if mode == "rev":
            if "f_struct" in d_outputs:
                if "f_aero" in d_inputs:
                    for imode in range(self.nmodes):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_inputs["f_aero"][3 * inode + k] += (
                                    self.mdisp[imode, inode, k]
                                    * d_outputs["f_struct"][imode]
                                )


class ModalDisplacements(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("mdisp")

    def setup(self):
        self.mdisp = self.options["mdisp"]
        self.nmodes = self.mdisp.shape[0]
        self.nnodes = self.mdisp.shape[1]

        self.add_input(
            "u_struct",
            shape_by_conn=True,
            tags=["mphys_coupling"],
            desc="modal displacement",
        )
        self.add_output(
            "u_aero",
            shape=self.nnodes * 3,
            tags=["mphys_coupling"],
            desc="nodal displacement",
        )

    def compute(self, inputs, outputs):
        outputs["u_aero"] = 0.0
        for imode in range(self.nmodes):
            for inode in range(self.nnodes):
                for k in range(3):
                    outputs["u_aero"][3 * inode + k] += (
                        self.mdisp[imode, inode, k] * inputs["u_struct"][imode]
                    )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            if "u_aero" in d_outputs:
                if "u_struct" in d_inputs:
                    for imode in range(self.nmodes):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_outputs["u_aero"][3 * inode + k] += (
                                    self.mdisp[imode, inode, k]
                                    * d_inputs["u_struct"][imode]
                                )
        if mode == "rev":
            if "u_aero" in d_outputs:
                if "u_struct" in d_inputs:
                    for imode in range(self.options["nmodes"]):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_inputs["u_struct"][imode] += (
                                    self.mdisp[imode, inode, k]
                                    * d_outputs["u_aero"][3 * inode + k]
                                )


class ModalXferBuilder(TimeDomainBuilder):
    def __init__(
        self,
        root_name,
        nmodes,
    ):
        self.root_name = root_name
        self.nmodes = nmodes
        self._read_mode_shapes()

    def _read_mode_shapes(self):
        for imode in range(self.nmodes):
            filename = f"{self.root_name}_mode{imode+1}.dat"
            fh = open(filename)
            while True:
                line = fh.readline()
                if "zone" in line.lower():
                    self.nnodes = int(line.split("=")[2].split(",")[0])
                    if imode == 0:
                        self.mdisp = np.zeros((self.nmodes, self.nnodes, 3))
                    for inode in range(self.nnodes):
                        line = fh.readline()
                        self.mdisp[imode, inode, 0] = float(line.split()[4])
                        self.mdisp[imode, inode, 1] = float(line.split()[5])
                        self.mdisp[imode, inode, 2] = float(line.split()[6])
                if not line:
                    break
            fh.close()

    def get_coupling_group_subsystem(self, scenario_name=None):
        return (
            ModalDisplacements(mdisp=self.mdisp),
            ModalForces(mdisp=self.mdisp),
        )
