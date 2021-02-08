import numpy as np
import openmdao.api as om

class SparDepth(om.ExplicitComponent):

    def initialize(self):
        
        self.options.declare('N_nodes', types=int)
        self.options.declare('elements', types=np.ndarray)
        self.options.declare('prop_ID', types=np.ndarray)
        self.options.declare('patches')
        
        self.N_sensor = 5
        self.KS_p = 50.
    
        self.under_complex_step = False
        
    def setup(self):

        self.add_input('x',np.zeros(self.options['N_nodes']*3))
        self.add_output('spar_depth',0.0)
        
    def compute(self,inputs,outputs):
        
        quad = self.options['elements']
        
        ## unpack coordinates
        
        X = inputs['x'][0::3]
        Y = inputs['x'][1::3]
        Z = inputs['x'][2::3]
        
        ## find nodes along the top and bottom of the rear spar

        US = self._component_elements(self.options['patches'].upper_skin,self.options['prop_ID'])
        US = np.unique(quad[US].flatten())
        
        LS = self._component_elements(self.options['patches'].lower_skin,self.options['prop_ID'])
        LS = np.unique(quad[LS].flatten())
        
        TE = self._component_elements(self.options['patches'].te_spar,self.options['prop_ID'])
        TE = np.unique(quad[TE].flatten())
        
        top_rear = np.intersect1d(US,TE)
        bot_rear = np.intersect1d(LS,TE)

        i = np.argsort(Y[top_rear])
        self.top_rear = top_rear[i]
        
        i = np.argsort(Y[bot_rear])
        self.bot_rear = bot_rear[i]        

        ## set up sensor points down the wing
        
        self.y = np.min(Y[self.top_rear]) + np.linspace(0,self.N_sensor,self.N_sensor+1)*(np.max(Y[self.top_rear])-np.min(Y[self.top_rear]))/self.N_sensor

        ## interpolate to get the z-coordinates at the sensor points
        
        self.z_up = self._interp(Y[self.top_rear],Z[self.top_rear],self.y)
        self.z_lo = self._interp(Y[self.bot_rear],Z[self.bot_rear],self.y)

        ## aggregate to compute the critica spar depth
        
        depth = self.z_up - self.z_lo
        
        outputs['spar_depth'] = 1/(np.log(np.sum(np.exp((1/depth)*self.KS_p)))/self.KS_p)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode): 
        
        ## unpack coordinates
        
        X = inputs['x'][0::3]
        Y = inputs['x'][1::3]
        Z = inputs['x'][2::3]
  
        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            
            ## derivative of sensor points with respect to Y
            
            y_Yderiv = np.zeros([len(self.y),len(Y)])

            j1 = np.argmin(Y[self.top_rear])
            j2 = np.argmax(Y[self.top_rear])
        
            y_Yderiv[:,self.top_rear[j1]] = np.linspace(1.,0.,self.N_sensor+1)
            y_Yderiv[:,self.top_rear[j2]] = np.linspace(0.,1.,self.N_sensor+1)
            
            ## derivative of z_up with respect to self.y, Y, and Z
            
            self.set_CS(True)
            
            z_up_y_deriv = np.zeros([len(self.z_up),len(self.y)])
            z_up_Y_deriv = np.zeros([len(self.z_up),len(Y)])
            z_up_Z_deriv = np.zeros([len(self.z_up),len(Z)])
            
            for i in range(0,len(self.y)):
                temp = self.y.astype('complex')
                temp[i] = temp[i] + 1j*1E-40
                temp = self._interp(Y[self.top_rear],Z[self.top_rear],temp)
                z_up_y_deriv[:,i] = np.imag(temp)/1E-40
            
            for i in range(0,len(self.top_rear)):
                temp = Y.astype('complex')
                temp[self.top_rear[i]] = temp[self.top_rear[i]] + 1j*1E-40
                temp = self._interp(temp[self.top_rear],Z[self.top_rear],self.y)
                z_up_Y_deriv[:,self.top_rear[i]] = np.imag(temp)/1E-40
                
                temp = Z.astype('complex')
                temp[self.top_rear[i]] = temp[self.top_rear[i]] + 1j*1E-40
                temp = self._interp(Y[self.top_rear],temp[self.top_rear],self.y)
                z_up_Z_deriv[:,self.top_rear[i]] = np.imag(temp)/1E-40
                
            ## derivative of z_lo with respect to self.y, Y, and Z

            z_lo_y_deriv = np.zeros([len(self.z_up),len(self.y)])
            z_lo_Y_deriv = np.zeros([len(self.z_up),len(Y)])
            z_lo_Z_deriv = np.zeros([len(self.z_up),len(Z)])
            
            for i in range(0,len(self.y)):
                temp = self.y.astype('complex')
                temp[i] = temp[i] + 1j*1E-40
                temp = self._interp(Y[self.bot_rear],Z[self.bot_rear],temp)
                z_lo_y_deriv[:,i] = np.imag(temp)/1E-40   
                
            for i in range(0,len(self.bot_rear)):    
                temp = Y.astype('complex')
                temp[self.bot_rear[i]] = temp[self.bot_rear[i]] + 1j*1E-40
                temp = self._interp(temp[self.bot_rear],Z[self.bot_rear],self.y)
                z_lo_Y_deriv[:,self.bot_rear[i]] = np.imag(temp)/1E-40               
                
                temp = Z.astype('complex')
                temp[self.bot_rear[i]] = temp[self.bot_rear[i]] + 1j*1E-40
                temp = self._interp(Y[self.bot_rear],temp[self.bot_rear],self.y)
                z_lo_Z_deriv[:,self.bot_rear[i]] = np.imag(temp)/1E-40
            
            self.set_CS(False)
            
            ## final derivative
            
            depth_Y = z_up_Y_deriv + z_up_y_deriv@y_Yderiv - z_lo_Y_deriv - z_lo_y_deriv@y_Yderiv
            depth_Z = z_up_Z_deriv - z_lo_Z_deriv
            
            depth = self.z_up - self.z_lo
            KS = 1/(np.log(np.sum(np.exp((1/depth)*self.KS_p)))/self.KS_p)
            
            KS_depth = np.exp((1/depth)*self.KS_p)/np.sum(np.exp((1/depth)*self.KS_p))
            KS_depth = (KS_depth*KS*KS)/depth/depth

            if 'spar_depth' in d_outputs:
                if 'x' in d_inputs:
                        
                    d_inputs['x'][1::3] += d_outputs['spar_depth']*(KS_depth@depth_Y)
                    d_inputs['x'][2::3] += d_outputs['spar_depth']*(KS_depth@depth_Z)
                    
            
        
    def set_CS(self,under_complex_step):
        self.under_complex_step = under_complex_step

    ## obtain the finite elements which are a member of a given component patch

    def _component_elements(self, patch,prop_ID):

        el = []
        for i in range(0,np.shape(patch)[0]):
            for j in range(0,np.shape(patch)[1]):
                el = np.r_[el,np.squeeze(np.argwhere(prop_ID==patch[i,j]-1))]
        
        el = el.astype(int)
            
        return el  

    ## linear interpolation of y vs x, to compute yq at xq locations

    def _interp(self, x, y, xq):
        
        if self.under_complex_step is True:
            x = x.astype('complex')
            y = y.astype('complex')
            xq = xq.astype('complex')
            
        yq = 0*xq
        
        for i in range(1,len(x)):
            if i == 1:
                a = np.argwhere(xq <= x[i])
            elif i == len(x)-1:
                a = np.argwhere(xq >= x[i-1])
            else:
                a = np.argwhere((xq <= x[i]) & (xq >= x[i-1]))
            yq[a] = y[i] - (y[i] - y[i-1])*(x[i] - xq[a])/(x[i] - x[i-1])
            
        return yq
                