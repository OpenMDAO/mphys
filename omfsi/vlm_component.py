from __future__ import division, print_function
import numpy as np

from openmdao.api import ImplicitComponent, ExplicitComponent, Group
from vlm_solver import VLM_solver, VLM_forces
from omfsi.assembler import OmfsiSolverAssembler

class VlmAssembler(OmfsiSolverAssembler):
   def __init__(self,options,comm):

      ## flow parameters

      self.comm = comm

      self.mach = options['mach']
      self.q_inf = options['q_inf']
      self.vel = options['vel']
      self.mu = options['mu']

      ## mesh parameters

      self.N_nodes = options['N_nodes']
      self.N_elements = options['N_elements']
      self.x_a0 = options['x_a0']
      self.quad = options['quad']

   def get_ndof(self):

      self.ndof = 3
      return self.ndof

   def get_nnodes(self):

      self.nnodes = self.N_nodes
      return self.nnodes

   def add_model_components(self,model,connection_srcs):

      model.add_subsystem('aero_mesh',VlmMesh(N_nodes=self.N_nodes, x_a0=self.x_a0))
      connection_srcs['x_a0'] = 'aero_mesh.x_a0_mesh'

   def add_scenario_components(self,model,scenario,connection_srcs):
      
      scenario.add_subsystem('aero_funcs',VlmOutput())

   def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):

      aero = Group()
      aero.add_subsystem('VLM_solver_comp', VLM_solver(N_nodes=self.N_nodes, N_elements=self.N_elements, quad=self.quad, mach=self.mach))
      aero.add_subsystem('VLM_forces_comp', VLM_forces(N_nodes=self.N_nodes, N_elements=self.N_elements, quad=self.quad, q_inf=self.q_inf, mach=self.mach, vel=self.vel, mu=self.mu))
      fsi_group.add_subsystem('aero',aero)

      connection_srcs['Cp'] = scenario.name+'.'+fsi_group.name+'.aero.VLM_solver_comp.Cp'
      connection_srcs['f_a'] = scenario.name+'.'+fsi_group.name+'.aero.VLM_forces_comp.fa'
      connection_srcs['CL'] = scenario.name+'.'+fsi_group.name+'.aero.VLM_forces_comp.CL'
      connection_srcs['CD'] = scenario.name+'.'+fsi_group.name+'.aero.VLM_forces_comp.CD'

   def connect_inputs(self,model,scenario,fsi_group,connection_srcs):

      model.connect(connection_srcs['alpha'],scenario.name+'.'+fsi_group.name+'.aero.VLM_solver_comp.alpha')

      model.connect(connection_srcs['x_a'],[scenario.name+'.'+fsi_group.name+'.aero.VLM_solver_comp.xa',
                                            scenario.name+'.'+fsi_group.name+'.aero.VLM_forces_comp.xa'])

      model.connect(connection_srcs['Cp'],scenario.name+'.'+fsi_group.name+'.aero.VLM_forces_comp.Cp')

      model.connect(connection_srcs['CL'],scenario.name+'.aero_funcs.CL')
      model.connect(connection_srcs['CD'],scenario.name+'.aero_funcs.CD')

class VlmMesh(ExplicitComponent):

   def initialize(self):

      self.options.declare('N_nodes', types=int)
      self.options.declare('x_a0', types=np.ndarray)

   def setup(self):

      N_nodes = self.options['N_nodes']
      self.add_output('x_a0_mesh',np.zeros(N_nodes*3))

   def compute(self,inputs,outputs):

      outputs['x_a0_mesh'] = self.options['x_a0']

class VlmOutput(ExplicitComponent):

   def setup(self):

      self.add_input('CL',0.0)
      self.add_input('CD',0.0)
      self.add_output('CL_out',0.0)
      self.add_output('CD_out',0.0)
      self.declare_partials('CL_out','CL')
      self.declare_partials('CL_out','CD')
      self.declare_partials('CD_out','CL')
      self.declare_partials('CD_out','CD')

   def compute(self,inputs,outputs):

      outputs['CL_out'] = inputs['CL']
      outputs['CD_out'] = inputs['CD']

   def compute_partials(self,inputs,partials):

      partials['CL_out','CL'] = 1.0
      partials['CL_out','CD'] = 0.0
      partials['CD_out','CL'] = 0.0
      partials['CD_out','CD'] = 1.0

