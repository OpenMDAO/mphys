import numpy as np
import openmdao.api as om
from mphys import Builder, MPhysVariables

from xfer import Xfer

X_STRUCT = MPhysVariables.Structures.COORDINATES
U_STRUCT = MPhysVariables.Structures.DISPLACEMENTS
F_STRUCT = MPhysVariables.Structures.Loads.AERODYNAMIC

X_AERO0 = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
U_AERO = MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS
F_AERO = MPhysVariables.Aerodynamics.Surface.LOADS

# EC which transfers displacements from structure to aero
class DispXfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.solver = self.options['solver']

        self.add_input(X_STRUCT, shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input(X_AERO0, shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input(U_STRUCT, shape_by_conn=True, distributed=True, tags=['mphys_coupling'])
        self.add_output(U_AERO, np.zeros(self.solver.aero.n_dof*self.solver.aero.n_nodes), distributed=True, tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        self.solver.xs = inputs[X_STRUCT]
        self.solver.xa = inputs[X_AERO0]
        self.solver.us = inputs[U_STRUCT]

        outputs[U_AERO] = self.solver.transfer_displacements()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            if U_AERO in d_outputs:
                d_xs, d_xa, d_us = self.solver.transfer_displacements_derivatives(
                    adjoint=d_outputs[U_AERO]
                )

                if X_STRUCT in d_inputs:
                    d_inputs[X_STRUCT] += d_xs
                if X_AERO0 in d_inputs:
                    d_inputs[X_AERO0] += d_xa
                if U_STRUCT in d_inputs:
                    d_inputs[U_STRUCT] += d_us


# EC which transfers loads from aero to structure
class LoadXfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.solver = self.options['solver']

        self.add_input(X_STRUCT, shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input(X_AERO0, shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input(F_AERO, shape_by_conn=True, distributed=True, tags=['mphys_coupling'])
        self.add_output(F_STRUCT, np.zeros(self.solver.struct.n_dof*self.solver.struct.n_nodes), distributed=True, tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        self.solver.xs = inputs[X_STRUCT]
        self.solver.xa = inputs[X_AERO0]
        self.solver.fa = inputs[F_AERO]

        outputs[F_STRUCT] = self.solver.transfer_loads()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            if F_STRUCT in d_outputs:
                d_xs, d_xa, d_fa = self.solver.transfer_loads_derivatives(
                    adjoint=d_outputs[F_STRUCT]
                )

                if X_STRUCT in d_inputs:
                    d_inputs[X_STRUCT] += d_xs
                if X_AERO0 in d_inputs:
                    d_inputs[X_AERO0] += d_xa
                if F_AERO in d_inputs:
                    d_inputs[F_AERO] += d_fa


# Builder
class XferBuilder(Builder):
    def __init__(self, aero_builder, struct_builder):
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder

    def initialize(self, comm):
        self.solver = Xfer(
            aero = self.aero_builder.solver,
            struct = self.struct_builder.solver,
            comm = comm
        )

    def get_coupling_group_subsystem(self, scenario_name=None):
        disp_xfer = DispXfer(solver=self.solver)
        load_xfer = LoadXfer(solver=self.solver)
        return disp_xfer, load_xfer
