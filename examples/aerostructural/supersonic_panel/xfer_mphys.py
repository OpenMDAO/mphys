import numpy as np
import openmdao.api as om
from xfer import Xfer

from mphys import Builder, MPhysVariables


# EC which transfers displacements from structure to aero
class DispXfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("solver")

        self.x_struct_name = MPhysVariables.Structures.COORDINATES
        self.x_aero0_name = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
        self.u_struct_name = MPhysVariables.Structures.DISPLACEMENTS
        self.u_aero_name = MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS

    def setup(self):
        self.solver = self.options["solver"]

        self.add_input(
            self.x_struct_name,
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input(
            self.x_aero0_name, shape_by_conn=True, distributed=True, tags=["mphys_coordinates"]
        )
        self.add_input(
            self.u_struct_name, shape_by_conn=True, distributed=True, tags=["mphys_coupling"]
        )
        self.add_output(
            self.u_aero_name,
            np.zeros(self.solver.aero.n_dof * self.solver.aero.n_nodes),
            distributed=True,
            tags=["mphys_coupling"],
        )

    def compute(self, inputs, outputs):
        self.solver.xs = inputs[self.x_struct_name]
        self.solver.xa = inputs[self.x_aero0_name]
        self.solver.us = inputs[self.u_struct_name]

        outputs[self.u_aero_name] = self.solver.transfer_displacements()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            if self.u_aero_name in d_outputs:
                d_xs, d_xa, d_us = self.solver.transfer_displacements_derivatives(
                    adjoint=d_outputs[self.u_aero_name]
                )

                if self.x_struct_name in d_inputs:
                    d_inputs[self.x_struct_name] += d_xs
                if self.x_aero0_name in d_inputs:
                    d_inputs[self.x_aero0_name] += d_xa
                if self.u_struct_name in d_inputs:
                    d_inputs[self.u_struct_name] += d_us


# EC which transfers loads from aero to structure
class LoadXfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("solver")

        self.x_aero0_name = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
        self.x_struct_name = MPhysVariables.Structures.COORDINATES
        self.f_aero_name = MPhysVariables.Aerodynamics.Surface.LOADS
        self.f_struct_name = MPhysVariables.Structures.Loads.AERODYNAMIC

    def setup(self):
        self.solver = self.options["solver"]

        self.add_input(
            self.x_struct_name,
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input(
            self.x_aero0_name, shape_by_conn=True, distributed=True, tags=["mphys_coordinates"]
        )
        self.add_input(
            self.f_aero_name, shape_by_conn=True, distributed=True, tags=["mphys_coupling"]
        )
        self.add_output(
            self.f_struct_name,
            np.zeros(self.solver.struct.n_dof * self.solver.struct.n_nodes),
            distributed=True,
            tags=["mphys_coupling"],
        )

    def compute(self, inputs, outputs):
        self.solver.xs = inputs[self.x_struct_name]
        self.solver.xa = inputs[self.x_aero0_name]
        self.solver.fa = inputs[self.f_aero_name]

        outputs[self.f_struct_name] = self.solver.transfer_loads()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            if self.f_struct_name in d_outputs:
                d_xs, d_xa, d_fa = self.solver.transfer_loads_derivatives(
                    adjoint=d_outputs[self.f_struct_name]
                )

                if self.x_struct_name in d_inputs:
                    d_inputs[self.x_struct_name] += d_xs
                if self.x_aero0_name in d_inputs:
                    d_inputs[self.x_aero0_name] += d_xa
                if self.f_aero_name in d_inputs:
                    d_inputs[self.f_aero_name] += d_fa


# Builder
class XferBuilder(Builder):
    def __init__(self, aero_builder, struct_builder):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder

    def initialize(self, comm):
        self.solver = Xfer(
            aero=self.aero_builder.solver, struct=self.struct_builder.solver, comm=comm
        )

    def get_coupling_group_subsystem(self, scenario_name=None):
        disp_xfer = DispXfer(solver=self.solver)
        load_xfer = LoadXfer(solver=self.solver)
        return disp_xfer, load_xfer
