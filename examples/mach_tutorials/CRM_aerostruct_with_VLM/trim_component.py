import numpy as np
import openmdao.api as om

class Trim(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('non_designable_weight', types=float)
        
        self.gravity = -9.81
        
    def setup(self):

        self.add_input('wing_area',0.0)
        self.add_input('CL',0.0)
        self.add_input('structural_mass',0.0)
        self.add_input('fuel_mass',0.0)
        self.add_input('q_inf',0.0)
        
        self.add_output('load_factor',0.0)
        
    def compute(self,inputs,outputs):
        
        self.L = inputs['CL']*inputs['wing_area']*inputs['q_inf']
        self.W = .5*self.options['non_designable_weight'] + self.gravity*(inputs['structural_mass'] + inputs['fuel_mass'])
        
        outputs['load_factor'] = self.L/self.W
           
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            if 'load_factor' in d_outputs:
                if 'wing_area' in d_inputs:
                    d_inputs['wing_area'] += d_outputs['load_factor']*inputs['CL']*inputs['q_inf']/self.W
                if 'CL' in d_inputs:
                    d_inputs['CL'] += d_outputs['load_factor']*inputs['wing_area']*inputs['q_inf']/self.W
                if 'structural_mass' in d_inputs:
                    d_inputs['structural_mass'] += d_outputs['load_factor']*-self.L*self.gravity/self.W/self.W
                if 'fuel_mass' in d_inputs:
                    d_inputs['fuel_mass'] += d_outputs['load_factor']*-self.L*self.gravity/self.W/self.W

class FuelMatch(om.ExplicitComponent):
    
    def initialize(self):
        
        self.gravity = -9.81
        self.reserve_fuel = 7500.
        
    def setup(self):
        
        self.add_input('fuel_mass',0.0)
        self.add_input('fuel_burn',0.0)
        self.add_input('fuel_DV',0.0)
        
        self.add_output('fuel_mismatch',0.0)
        
    def compute(self,inputs,outputs):
       
       outputs['fuel_mismatch'] = (inputs['fuel_burn']/self.gravity + self.reserve_fuel)/inputs['fuel_mass'] - inputs['fuel_DV']
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            if 'fuel_mismatch' in d_outputs:
                if 'fuel_mass' in d_inputs:
                    d_inputs['fuel_mass'] -= d_outputs['fuel_mismatch']*(inputs['fuel_burn']/self.gravity + self.reserve_fuel)/inputs['fuel_mass']/inputs['fuel_mass']
                if 'fuel_burn' in d_inputs:
                    d_inputs['fuel_burn'] += d_outputs['fuel_mismatch']/self.gravity/inputs['fuel_mass']
                if 'fuel_DV' in d_inputs:
                    d_inputs['fuel_DV'] -= d_outputs['fuel_mismatch']