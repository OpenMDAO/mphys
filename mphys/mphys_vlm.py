import numpy as np

import openmdao.api as om
from mphys import Builder

from vlm_solver import Vlm

class VlmMesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare('x_aero0')
    def setup(self):
        self.add_output('x_aero0', val=self.options['x_aero0'], tags=['mphys_coordinates'])

class VlmSolver(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('solver', recordable=False)
        self.options.declare('complex_step', default=False)

    def setup(self):
        self.vlm = self.options['solver']

        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('aoa',0., units = 'rad', tags=['mphys_input'])
        self.add_input('mach',0., tags=['mphys_input'])

        self.add_output('Cp',np.zeros(int(self.vlm.N_elements/2)))

        self.declare_partials('Cp','aoa')
        self.declare_partials('Cp','x_aero')
        self.declare_partials('Cp','Cp')

        self.set_check_partial_options(wrt='*',directional=True,method='cs')

    def solve_nonlinear(self,inputs,outputs):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.aoa = inputs['aoa']
        self.vlm.mach = inputs['mach']

        ## solve

        self.vlm.complex_step = self.options['complex_step']
        self.vlm.compute_AIC()
        self.AIC = self.vlm.AIC
        self.vlm.solve_system()
        self.vlm.complex_step = False

        outputs['Cp'] = self.vlm.Cp

    def apply_nonlinear(self,inputs,outputs,residuals):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.aoa = inputs['aoa']
        self.vlm.mach = inputs['mach']

        ## compute residual

        self.vlm.complex_step = self.options['complex_step']
        self.vlm.compute_AIC()
        self.vlm.complex_step = False

        residuals['Cp'] = np.dot(self.vlm.AIC,outputs['Cp']) - self.vlm.w

    def linearize(self,inputs,outputs,partials):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.aoa = inputs['aoa']
        self.vlm.mach = inputs['mach']
        self.vlm.Cp = outputs['Cp']

        ## compute derivatives

        self.vlm.compute_residual_derivatives()

        partials['Cp','aoa'] = np.ones(len(self.vlm.w))
        partials['Cp','x_aero'] = self.vlm.R_xa
        partials['Cp','Cp'] = self.AIC

    def solve_linear(self,d_outputs,d_residuals,mode):
        if mode == 'rev':
            d_residuals['Cp'] = np.linalg.solve(self.AIC.transpose(), d_outputs['Cp'])


class VlmForces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver', recordable=False)
        self.options.declare('complex_step', default=False)

    def setup(self):
        self.vlm = self.options['solver']

        self.add_input('q_inf', 0., tags=['mphys_input'])
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('Cp',np.zeros(int(self.vlm.N_elements/2)))

        self.add_output('f_aero', np.zeros(self.vlm.N_nodes*3), tags=['mphys_coupling'])

        self.declare_partials('f_aero','x_aero')
        self.declare_partials('f_aero','Cp')

        self.set_check_partial_options(wrt='*',directional=True,method='cs')

    def compute(self,inputs,outputs):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.q_inf = inputs['q_inf']
        self.vlm.Cp = inputs['Cp']

        ## compute forces

        self.vlm.complex_step = self.options['complex_step']
        self.vlm.compute_forces()
        self.vlm.complex_step = False

        outputs['f_aero'] = self.vlm.fa

    def compute_partials(self,inputs,partials):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.q_inf = inputs['q_inf']
        self.vlm.Cp = inputs['Cp']

        ## compute derivatives
        self.vlm.compute_shape_derivatives = True
        self.vlm.compute_forces()
        self.vlm.compute_shape_derivatives = False

        partials['f_aero','x_aero'] = self.vlm.fa_xa
        partials['f_aero','Cp'] = self.vlm.fa_Cp


## EC which computes the aero coefficients

class VlmCoefficients(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver', recordable=False)
        self.options.declare('complex_step', default=False)

    def setup(self):
        self.vlm = self.options['solver']

        self.add_input('mach', 0., tags=['mphys_input'])
        self.add_input('vel', 0., tags=['mphys_input'])
        self.add_input('nu', 0., tags=['mphys_input'])
        self.add_input('x_aero', shape_by_conn=True, tags=['mphys_coupling'])
        self.add_input('Cp',np.zeros(int(self.vlm.N_elements/2)))

        self.add_output('C_L', tags=['mphys_result'])
        self.add_output('C_D', tags=['mphys_result'])

        self.declare_partials('C_L','x_aero')
        self.declare_partials('C_D','x_aero')
        self.declare_partials('C_L','Cp')
        self.declare_partials('C_D','Cp')

        self.set_check_partial_options(wrt='*',directional=True,method='cs')

    def compute(self,inputs,outputs):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.mach = inputs['mach']
        self.vlm.vel = inputs['vel']
        self.vlm.nu = inputs['nu']
        self.vlm.Cp = inputs['Cp']

        ## compute coefficients

        self.vlm.complex_step = self.options['complex_step']
        self.vlm.compute_coefficients()
        self.vlm.print_results()
        self.vlm.write_solution_file("VLM_output.dat")
        self.vlm.complex_step = False

        outputs['C_L'] = self.vlm.CL
        outputs['C_D'] = self.vlm.CD

    def compute_partials(self,inputs,partials):

        ## update VLM object

        self.vlm.set_mesh_coordinates(inputs['x_aero'])
        self.vlm.mach = inputs['mach']
        self.vlm.vel = inputs['vel']
        self.vlm.nu = inputs['nu']
        self.vlm.Cp = inputs['Cp']

        ## compute derivatives

        self.vlm.compute_shape_derivatives = True
        self.vlm.compute_coefficients()
        self.vlm.compute_shape_derivatives = False

        partials['C_L','x_aero'] = self.vlm.CL_xa
        partials['C_D','x_aero'] = self.vlm.CD_xa
        partials['C_L','Cp'] = self.vlm.CL_Cp
        partials['C_D','Cp'] = self.vlm.CD_Cp


class VlmGroup(om.Group):
    def initialize(self):
        self.options.declare('solver')
        self.options.declare('complex_step', default=False)

    def setup(self):
        self.solver = self.options['solver']
        complex_step = self.options['complex_step']

        self.add_subsystem('aero_solver', VlmSolver(
            solver=self.solver, complex_step = complex_step),
            promotes_inputs=['aoa','mach','x_aero'],
            promotes_outputs=['Cp']
            )

        self.add_subsystem('aero_forces', VlmForces(
            solver=self.solver, complex_step=complex_step),
            promotes_inputs=['q_inf','Cp','x_aero'],
            promotes_outputs=['f_aero']
            )

        self.add_subsystem('aero_coefficients', VlmCoefficients(
            solver=self.solver, complex_step=complex_step),
            promotes_inputs=['mach','vel','nu','Cp','x_aero'],
            promotes_outputs=['C_L','C_D']
            )


class VlmBuilder(Builder):
    def __init__(self, meshfile, compute_traction=False, complex_step=False):
        self.meshfile = meshfile
        self.compute_traction = compute_traction
        self.complex_step = complex_step

    def initialize(self, comm):
        self.solver = Vlm(compute_traction=self.compute_traction)
        self.solver.read_mesh(self.meshfile)
        self.x_aero0 = self.solver.xa

    def get_mesh_coordinate_subsystem(self):
        return VlmMesh(x_aero0=self.x_aero0)

    def get_coupling_group_subsystem(self):
        return VlmGroup(solver=self.solver, complex_step=self.complex_step)

    def get_number_of_nodes(self):
        return self.x_aero0.size // 3

    def get_ndof(self):
        return self.x_aero0.size // 3
