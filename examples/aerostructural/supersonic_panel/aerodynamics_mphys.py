import numpy as np
import openmdao.api as om
from mphys import Builder, MPhysVariables
from mpi4py import MPI

from piston_theory import PistonTheory

# IVC which returns a baseline mesh
class AeroMesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare('x_aero0')
    def setup(self):
        self.x_aero0_name = MPhysVariables.coordinates_initial_aerodynamic
        self.add_output(self.x_aero0_name, val=self.options['x_aero0'], distributed=True, tags=['mphys_coordinates'])


# IC which computes aero pressures
class AeroSolver(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('solver')

        self.x_aero_name = MPhysVariables.coordinates_deformed_aerodynamic

    def setup(self):
        self.solver = self.options['solver']

        self.add_input(self.x_aero_name, shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('aoa', 0., units = 'deg', tags=['mphys_input'])
        self.add_input('qdyn', 0., tags=['mphys_input'])
        self.add_input('mach', 0., tags=['mphys_input'])

        # pressure output will only exist on rank 0
        n = self.comm.allreduce(self.solver.n_nodes, op=MPI.SUM)
        if self.comm.Get_rank() == 0:
            self.add_output('pressure', np.zeros(n-1), distributed=True, tags=['mphys_coupling'])
        else:
            self.add_output('pressure', np.zeros(0), distributed=True, tags=['mphys_coupling'])

    def solve_nonlinear(self,inputs,outputs):

        self.solver.xyz = inputs[self.x_aero_name]
        self.solver.aoa = inputs['aoa']
        self.solver.qdyn = inputs['qdyn']
        self.solver.mach = inputs['mach']

        outputs['pressure'] = self.solver.compute_pressure()

    def apply_nonlinear(self,inputs,outputs,residuals):
        self.solver.xyz = inputs[self.x_aero_name]
        self.solver.aoa = inputs['aoa']
        self.solver.qdyn = inputs['qdyn']
        self.solver.mach = inputs['mach']

        residuals['pressure'] = self.solver.compute_residual(
            pressure=outputs['pressure']
        )

    def solve_linear(self,d_outputs,d_residuals,mode):
        if mode == 'rev':
            d_residuals['pressure'] = d_outputs['pressure']

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        if mode == 'rev':
            if 'pressure' in d_residuals:
                if 'pressure' in d_outputs:
                    d_outputs['pressure'] += d_residuals['pressure']

                d_xa, d_aoa, d_qdyn, d_mach = self.solver.compute_pressure_derivatives(
                    adjoint=d_residuals['pressure']
                )

                if self.x_aero_name in d_inputs:
                    d_inputs[self.x_aero_name] += d_xa
                if 'aoa' in d_inputs:
                    d_inputs['aoa'] += d_aoa
                if 'qdyn' in d_inputs:
                    d_inputs['qdyn'] += d_qdyn
                if 'mach' in d_inputs:
                    d_inputs['mach'] += d_mach


# EC which computes aero forces
class AeroForces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.x_aero_name = MPhysVariables.coordinates_deformed_aerodynamic
        self.f_aero_name = MPhysVariables.loads_aerodynamic

        self.solver = self.options['solver']

        self.add_input(self.x_aero_name, shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('pressure', shape_by_conn=True, distributed=True, tags=['mphys_coupling'])
        self.add_output(self.f_aero_name, np.zeros(self.solver.n_nodes*self.solver.n_dof), distributed=True, tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        self.solver.xyz = inputs[self.x_aero_name]
        self.solver.pressure = inputs['pressure']

        outputs[self.f_aero_name] = self.solver.compute_force()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            if self.f_aero_name in d_outputs:
                d_xa, d_p = self.solver.compute_force_derivatives(
                    adjoint=d_outputs[self.f_aero_name]
                )

                if self.x_aero_name in d_inputs:
                    d_inputs[self.x_aero_name] += d_xa
                if 'pressure' in d_inputs:
                    d_inputs['pressure'] += d_p


# EC which computes the aero function
class AeroFunction(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.solver = self.options['solver']

        self.add_input('pressure', shape_by_conn=True, distributed=True, tags=['mphys_coupling'])
        self.add_input('qdyn', 0., tags=['mphys_input'])
        self.add_output('C_L', tags=['mphys_result'])

    def compute(self,inputs,outputs):
        self.solver.qdyn = inputs['qdyn']
        self.solver.pressure = inputs['pressure']

        outputs['C_L'] = self.solver.compute_lift()

        self.solver.write_output()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'rev':
            if 'C_L' in d_outputs:
                d_p, d_qdyn = self.solver.compute_lift_derivatives(
                    adjoint=d_outputs['C_L']
                )

                if 'pressure' in d_inputs:
                    d_inputs['pressure'] += d_p
                if 'qdyn' in d_inputs:
                    d_inputs['qdyn'] += d_qdyn


# Group which holds the solver and force computation
class AeroSolverGroup(om.Group):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.add_subsystem('aero_solver', AeroSolver(
            solver = self.options['solver']),
            promotes=['*']
        )

        self.add_subsystem('aero_forces', AeroForces(
            solver = self.options['solver']),
            promotes=['*']
        )


# Group which holds the function computation
class AeroFunctionGroup(om.Group):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.add_subsystem('aero_function', AeroFunction(
            solver = self.options['solver']),
            promotes=['*']
        )


# Builder
class AeroBuilder(Builder):
    def __init__(self, options):
        self.options = options

    def initialize(self, comm):
        self.solver = PistonTheory(
            panel_chord=self.options['panel_chord'],
            panel_width=self.options['panel_width'],
            N_el=self.options['N_el'],
            comm=comm
        )

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        if self.solver.owned is not None:
            x_aero0 = np.c_[self.solver.x,self.solver.y,self.solver.z].flatten(order='C')
        else:
            x_aero0 = np.zeros(0)
        return AeroMesh(x_aero0=x_aero0)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return AeroSolverGroup(solver=self.solver)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return AeroFunctionGroup(solver=self.solver)

    def get_number_of_nodes(self):
        return self.solver.n_nodes

    def get_ndof(self):
        return self.soler.n_dof
