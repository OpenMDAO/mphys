import numpy as np
import openmdao.api as om

class FlightMetrics(om.ExplicitComponent):
    
    def initialize(self):
    
        self.options.declare('non_designable_weight', types=float)
        self.options.declare('range', types=float)
        self.options.declare('TSFC', types=float)
        self.options.declare('beta', types=float)        
        self.options.declare('gravity', types=float)

    def setup(self):

        self.add_input('CL',0.0)
        self.add_input('CD',0.0)
        self.add_input('structural_mass',0.0)
        self.add_input('fuel_mass',0.0)
        self.add_input('velocity',0.0)
        
        self.add_output('FB',0.0)
        self.add_output('LGW',0.0)
        self.add_output('final_objective',0.0)
        
    def compute(self,inputs,outputs):
        
        self.TOGW = .5*self.options['non_designable_weight'] + self.options['gravity']*(inputs['structural_mass'] + inputs['fuel_mass'])
    
        self.L_D = inputs['CL']/inputs['CD']
        
        outputs['FB'] = (1-np.exp(-(self.options['range']*self.options['TSFC'])/inputs['velocity']/self.L_D))*self.TOGW
        
        outputs['LGW'] = self.TOGW - outputs['FB']
    
        outputs['final_objective'] = (outputs['FB']*self.options['beta'] + outputs['LGW']*(1-self.options['beta']))*2/self.options['gravity']/1E5
               
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        FB_CL = -(inputs['CD']*self.options['range']*self.TOGW*self.options['TSFC']*np.exp(-(inputs['CD']*self.options['range']*self.options['TSFC'])/(inputs['CL']*inputs['velocity'])))/(inputs['CL']**2*inputs['velocity'])
        FB_CD = (self.options['range']*self.TOGW*self.options['TSFC']*np.exp(-(inputs['CD']*self.options['range']*self.options['TSFC'])/(inputs['CL']*inputs['velocity'])))/(inputs['CL']*inputs['velocity'])
        FB_TOGW = 1 - np.exp(-(inputs['CD']*self.options['range']*self.options['TSFC'])/(inputs['CL']*inputs['velocity']))
        
        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            if 'FB' in d_outputs:
                if 'CL' in d_inputs:
                    d_inputs['CL'] += d_outputs['FB']*FB_CL
                if 'CD' in d_inputs:
                    d_inputs['CD'] += d_outputs['FB']*FB_CD
                if 'structural_mass' in d_inputs:
                    d_inputs['structural_mass'] += d_outputs['FB']*FB_TOGW*self.options['gravity']
                if 'fuel_mass' in d_inputs:
                    d_inputs['fuel_mass'] += d_outputs['FB']*FB_TOGW*self.options['gravity']
            if 'LGW' in d_outputs:         
                if 'CL' in d_inputs:
                    d_inputs['CL'] += d_outputs['LGW']*-FB_CL
                if 'CD' in d_inputs:
                    d_inputs['CD'] += d_outputs['LGW']*-FB_CD
                if 'structural_mass' in d_inputs:
                    d_inputs['structural_mass'] += d_outputs['LGW']*(1-FB_TOGW)*self.options['gravity'] 
                if 'fuel_mass' in d_inputs:
                    d_inputs['fuel_mass'] += d_outputs['LGW']*(1-FB_TOGW)*self.options['gravity']   
            if 'final_objective' in d_outputs:         
                if 'CL' in d_inputs:
                    d_inputs['CL'] += d_outputs['final_objective']*(FB_CL*self.options['beta'] + -FB_CL*(1-self.options['beta']))*2/self.options['gravity']/1E5
                if 'CD' in d_inputs:
                    d_inputs['CD'] += d_outputs['final_objective']*(FB_CD*self.options['beta'] + -FB_CD*(1-self.options['beta']))*2/self.options['gravity']/1E5
                if 'structural_mass' in d_inputs:
                    d_inputs['structural_mass'] += d_outputs['final_objective']*(FB_TOGW*self.options['beta'] + (1-FB_TOGW)*(1-self.options['beta']))*2/1E5
                if 'fuel_mass' in d_inputs:
                    d_inputs['fuel_mass'] += d_outputs['final_objective']*(FB_TOGW*self.options['beta'] + (1-FB_TOGW)*(1-self.options['beta']))*2/1E5                                  
                    

                    
