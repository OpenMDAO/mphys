import numpy as np
from mpi4py import MPI
from openmdao.api import ImplicitComponent
from dafoam import PYDAFOAM


class DAFoamSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the DAFoam flow and adjoint solvers
    """

    def __init__(self, options):

        # options dictionary for DAFoam
        self.options = options

    # api level method for all builders
    def initialize(self, comm):
        self.solver = PYDAFOAM(options=self.options, comm=comm)

    def setup(self):
        pass

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass

    def solve_nonlinear(self, inputs, outputs):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        pass

    def solve_linear(self, d_outputs, d_residuals, mode):
        pass