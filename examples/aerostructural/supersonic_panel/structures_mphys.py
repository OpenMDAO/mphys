import numpy as np
import openmdao.api as om
from beam_solver import Beam

from mphys import Builder


# IVC which returns a baseline mesh
class StructMesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare("x_struct0")

    def setup(self):
        self.add_output(
            "x_struct0",
            val=self.options["x_struct0"],
            distributed=True,
            tags=["mphys_coordinates"],
        )


# IC which computes structural displacements
class StructSolver(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("solver")

    def setup(self):
        self.solver = self.options["solver"]

        self.add_input("dv_struct", shape_by_conn=True, tags=["mphys_input"])
        self.add_input(
            "x_struct0",
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "f_struct", shape_by_conn=True, distributed=True, tags=["mphys_coupling"]
        )
        self.add_input("modulus", 0.0, tags=["mphys_input"])
        self.add_output(
            "u_struct",
            np.zeros(self.solver.n_dof * self.solver.n_nodes),
            distributed=True,
            tags=["mphys_coupling"],
        )

        # bc correct term, needed for LNBGS residuals to properly converge to 0
        self.bc_correct = np.zeros(self.solver.n_dof * self.solver.n_nodes)

    def solve_nonlinear(self, inputs, outputs):
        self.solver.dv_struct = inputs["dv_struct"]
        self.solver.xyz = inputs["x_struct0"]
        self.solver.modulus = inputs["modulus"]

        outputs["u_struct"] = self.solver.solve_system(f=inputs["f_struct"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        self.solver.dv_struct = inputs["dv_struct"]
        self.solver.xyz = inputs["x_struct0"]
        self.solver.modulus = inputs["modulus"]

        residuals["u_struct"] = self.solver.compute_residual(
            u=outputs["u_struct"], f=inputs["f_struct"]
        )

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "rev":
            d_residuals["u_struct"] = self.solver.solve_system(f=d_outputs["u_struct"])

            # correct the boundary condition dof, in order to set the LHS equal to the RHS
            self.bc_correct = self.solver.bc_correction(u=d_outputs["u_struct"])
            d_residuals["u_struct"] += self.bc_correct

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode == "rev":
            if "u_struct" in d_residuals:
                self.solver.dv_struct = inputs["dv_struct"]
                self.solver.xyz = inputs["x_struct0"]
                self.solver.modulus = inputs["modulus"]

                adjoint = self.solver.set_adjoint(adjoint=d_residuals["u_struct"])

                if "u_struct" in d_outputs:
                    d_outputs["u_struct"] += self.solver.compute_residual(
                        u=adjoint, f=np.zeros_like(adjoint)
                    )

                    # add back in non-zero values at the bc DOFs, so the LNBGS residuals look correct
                    d_outputs["u_struct"] += self.bc_correct

                (
                    d_dv_struct,
                    d_xs,
                    d_modulus,
                ) = self.solver.compute_stiffness_derivatives(
                    u=outputs["u_struct"], adjoint=adjoint
                )

                if "dv_struct" in d_inputs:
                    d_inputs["dv_struct"] += d_dv_struct
                if "x_struct0" in d_inputs:
                    d_inputs["x_struct0"] += d_xs
                if "f_struct" in d_inputs:
                    d_inputs["f_struct"] -= adjoint
                if "modulus" in d_inputs:
                    d_inputs["modulus"] += d_modulus


# EC which computes the structural function
class StructFunction(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("solver")

    def setup(self):
        self.solver = self.options["solver"]

        self.aggregation_parameter = 20.0

        self.add_input("dv_struct", shape_by_conn=True, tags=["mphys_input"])
        self.add_input(
            "x_struct0",
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input(
            "u_struct", shape_by_conn=True, distributed=True, tags=["mphys_coupling"]
        )
        self.add_input("modulus", 0.0, tags=["mphys_input"])
        self.add_input("yield_stress", 0.0, tags=["mphys_input"])
        self.add_output("func_struct", 0.0, tags=["mphys_result"])

    def compute(self, inputs, outputs):
        self.solver.dv_struct = inputs["dv_struct"]
        self.solver.xyz = inputs["x_struct0"]
        self.solver.modulus = inputs["modulus"]
        self.solver.yield_stress = inputs["yield_stress"]

        self.stress, outputs["func_struct"] = self.solver.compute_stress(
            u=inputs["u_struct"], aggregation_parameter=self.aggregation_parameter
        )

        self.solver.write_output(u=inputs["u_struct"], stress=self.stress)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            if "func_struct" in d_outputs:
                self.solver.dv_struct = inputs["dv_struct"]
                self.solver.xyz = inputs["x_struct0"]
                self.solver.modulus = inputs["modulus"]
                self.solver.yield_stress = inputs["yield_stress"]

                (
                    d_dv_struct,
                    d_xs,
                    d_us,
                    d_modulus,
                    d_yield_stress,
                ) = self.solver.compute_stress_derivatives(
                    u=inputs["u_struct"],
                    stress=self.stress,
                    aggregation_parameter=self.aggregation_parameter,
                    adjoint=d_outputs["func_struct"],
                )

                if "dv_struct" in d_inputs:
                    d_inputs["dv_struct"] += d_dv_struct
                if "x_struct0" in d_inputs:
                    d_inputs["x_struct0"] += d_xs
                if "u_struct" in d_inputs:
                    d_inputs["u_struct"] += d_us
                if "modulus" in d_inputs:
                    d_inputs["modulus"] += d_modulus
                if "yield_stress" in d_inputs:
                    d_inputs["yield_stress"] += d_yield_stress


# EC which computes the structural masss
class StructMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("solver")

    def setup(self):
        self.solver = self.options["solver"]

        self.add_input("dv_struct", shape_by_conn=True, tags=["mphys_input"])
        self.add_input(
            "x_struct0",
            shape_by_conn=True,
            distributed=True,
            tags=["mphys_coordinates"],
        )
        self.add_input("density", 0.0, tags=["mphys_input"])
        self.add_output("mass", tags=["mphys_result"])

    def compute(self, inputs, outputs):
        self.solver.dv_struct = inputs["dv_struct"]
        self.solver.xyz = inputs["x_struct0"]
        self.solver.density = inputs["density"]

        outputs["mass"] = self.solver.compute_mass()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "rev":
            if "mass" in d_outputs:
                self.solver.dv_struct = inputs["dv_struct"]
                self.solver.xyz = inputs["x_struct0"]
                self.solver.density = inputs["density"]

                d_dv_struct, d_xs, d_density = self.solver.compute_mass_derivatives(
                    adjoint=d_outputs["mass"]
                )

                if "dv_struct" in d_inputs:
                    d_inputs["dv_struct"] += d_dv_struct
                if "x_struct0" in d_inputs:
                    d_inputs["x_struct0"] += d_xs
                if "density" in d_inputs:
                    d_inputs["density"] += d_density


# Group which holds the solver
class StructSolverGroup(om.Group):
    def initialize(self):
        self.options.declare("solver")

    def setup(self):
        self.add_subsystem(
            "struct_solver", StructSolver(solver=self.options["solver"]), promotes=["*"]
        )


# Group which holds the function computation
class StructFunctionGroup(om.Group):
    def initialize(self):
        self.options.declare("solver")

    def setup(self):
        self.add_subsystem(
            "struct_function",
            StructFunction(solver=self.options["solver"]),
            promotes=["*"],
        )

        self.add_subsystem(
            "struct_mass", StructMass(solver=self.options["solver"]), promotes=["*"]
        )


# Builder
class StructBuilder(Builder):
    def __init__(self, options):
        self.options = options

    def initialize(self, comm):
        self.solver = Beam(
            panel_chord=self.options["panel_chord"],
            panel_width=self.options["panel_width"],
            N_el=self.options["N_el"],
            comm=comm,
        )

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        if self.solver.owned is not None:
            x_struct0 = np.c_[self.solver.x, self.solver.y, self.solver.z].flatten(
                order="C"
            )
        else:
            x_struct0 = np.zeros(0)
        return StructMesh(x_struct0=x_struct0)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return StructSolverGroup(solver=self.solver)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return StructFunctionGroup(solver=self.solver)

    def get_number_of_nodes(self):
        return self.solver.n_nodes

    def get_ndof(self):
        return self.solver.n_dof