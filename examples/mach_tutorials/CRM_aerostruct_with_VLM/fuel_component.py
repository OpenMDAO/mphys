import numpy as np
import openmdao.api as om

## compute the available fuel mass inside the wingbox

class FuelMass:
    
    def __init__(self,nodes,quads,prop_ID,patches,fuel_density):
        
        self.nodes = nodes
        self.quad = quads
        self.prop_ID = prop_ID
        self.patches = patches
        self.fuel_density = fuel_density

        self.N_span = 15
        self.volume_knock_down = 0.85
        
        self.under_complex_step = False
        
    def set_CS(self,under_complex_step):
        self.under_complex_step = under_complex_step
        
    def compute(self):
        
        ## unpack coordinates
        
        X = self.nodes[0::3]
        Y = self.nodes[1::3]
        Z = self.nodes[2::3]
        
        ## compute spanwise subdivisions
        
        self.y, self.y_Yderiv = self._span_division(Y)
        
        ## find elements of each component
        
        US = _component_elements(self.patches.upper_skin,self.prop_ID)
        LS = _component_elements(self.patches.lower_skin,self.prop_ID)
        
        ## loop over each span station
        
        if self.under_complex_step is True:
            self.x = np.zeros(self.N_span,dtype=complex)
            self.z = np.zeros(self.N_span,dtype=complex)
            self.mass = np.zeros(self.N_span,dtype=complex)
        else:
            self.x = np.zeros(self.N_span) 
            self.z = np.zeros(self.N_span)
            self.mass = np.zeros(self.N_span)
            
        self.x_Xderiv = np.zeros([len(self.x),len(X)])
        self.z_Zderiv = np.zeros([len(self.z),len(Z)])
          
        self.mass_Xderiv = np.zeros([self.N_span,len(Y)])
        self.mass_Yderiv = np.zeros([self.N_span,len(Y)])
        self.mass_Zderiv = np.zeros([self.N_span,len(Y)])
        
        for i in range(0,self.N_span):
            
            ## find upper skin elements in this span station
            
            a = US[np.where(np.logical_and(np.mean(Y[self.quad[US,:]],axis=1)<=self.y[i+1],np.mean(Y[self.quad[US,:]],axis=1)>self.y[i]))]
            
            self.x[i] = np.mean(X[self.quad[a,:]])
            self.z[i] = np.mean(Z[self.quad[a,:]])
            
            ## loop over elements
            
            for j in range(0,len(a)):
                
                self.x_Xderiv[i,self.quad[a[j],:]] = self.x_Xderiv[i,self.quad[a[j],:]] + 1/2/len(a)/4
                self.z_Zderiv[i,self.quad[a[j],:]] = self.z_Zderiv[i,self.quad[a[j],:]] + 1/2/len(a)/4
                
                # compute element area
            
                X1 = X[self.quad[a[j],0]]; X2 = X[self.quad[a[j],1]]; X3 = X[self.quad[a[j],2]]; X4 = X[self.quad[a[j],3]];
                Y1 = Y[self.quad[a[j],0]]; Y2 = Y[self.quad[a[j],1]]; Y3 = Y[self.quad[a[j],2]]; Y4 = Y[self.quad[a[j],3]];
                Z1 = Z[self.quad[a[j],0]]; Z2 = Z[self.quad[a[j],1]]; Z3 = Z[self.quad[a[j],2]]; Z4 = Z[self.quad[a[j],3]];
    
                A, A_X, A_Y, A_Z = self._element_area(X1,X2,X3,X4,Y1,Y2,Y3,Y4,Z1,Z2,Z3,Z4)
                
                ## compute fuel mass
                
                self.mass[i] = self.mass[i] + self.volume_knock_down*self.fuel_density*A*(Z1+Z2+Z3+Z4)/4
            
                ## geometry derivatives
                
                self.mass_Xderiv[i,self.quad[a[j],:]] = self.mass_Xderiv[i,self.quad[a[j],:]] + self.volume_knock_down*self.fuel_density*A_X*(Z1+Z2+Z3+Z4)/4
                self.mass_Yderiv[i,self.quad[a[j],:]] = self.mass_Yderiv[i,self.quad[a[j],:]] + self.volume_knock_down*self.fuel_density*A_Y*(Z1+Z2+Z3+Z4)/4
                self.mass_Zderiv[i,self.quad[a[j],:]] = self.mass_Zderiv[i,self.quad[a[j],:]] + self.volume_knock_down*self.fuel_density*A_Z*(Z1+Z2+Z3+Z4)/4
                self.mass_Zderiv[i,self.quad[a[j],:]] = self.mass_Zderiv[i,self.quad[a[j],:]] + self.volume_knock_down*self.fuel_density*A*np.ones(4)/4
        
            ## find lower skin elements in this span station
            
            a = LS[np.where(np.logical_and(np.mean(Y[self.quad[LS,:]],axis=1)<=self.y[i+1],np.mean(Y[self.quad[LS,:]],axis=1)>self.y[i]))]
            
            self.x[i] = (self.x[i] + np.mean(X[self.quad[a,:]]))/2
            self.z[i] = (self.z[i] + np.mean(Z[self.quad[a,:]]))/2
            
            ## loop over elements
            
            for j in range(0,len(a)):
                
                self.x_Xderiv[i,self.quad[a[j],:]] = self.x_Xderiv[i,self.quad[a[j],:]] + 1/2/len(a)/4
                self.z_Zderiv[i,self.quad[a[j],:]] = self.z_Zderiv[i,self.quad[a[j],:]] + 1/2/len(a)/4
                
                # compute element area
            
                X1 = X[self.quad[a[j],0]]; X2 = X[self.quad[a[j],1]]; X3 = X[self.quad[a[j],2]]; X4 = X[self.quad[a[j],3]];
                Y1 = Y[self.quad[a[j],0]]; Y2 = Y[self.quad[a[j],1]]; Y3 = Y[self.quad[a[j],2]]; Y4 = Y[self.quad[a[j],3]];
                Z1 = Z[self.quad[a[j],0]]; Z2 = Z[self.quad[a[j],1]]; Z3 = Z[self.quad[a[j],2]]; Z4 = Z[self.quad[a[j],3]];
    
                A, A_X, A_Y, A_Z = self._element_area(X1,X2,X3,X4,Y1,Y2,Y3,Y4,Z1,Z2,Z3,Z4)
                
                ## compute fuel mass
                
                self.mass[i] = self.mass[i] - self.volume_knock_down*self.fuel_density*A*(Z1+Z2+Z3+Z4)/4
                
                ## geometry derivatives
                
                self.mass_Xderiv[i,self.quad[a[j],:]] = self.mass_Xderiv[i,self.quad[a[j],:]] - self.volume_knock_down*self.fuel_density*A_X*(Z1+Z2+Z3+Z4)/4
                self.mass_Yderiv[i,self.quad[a[j],:]] = self.mass_Yderiv[i,self.quad[a[j],:]] - self.volume_knock_down*self.fuel_density*A_Y*(Z1+Z2+Z3+Z4)/4
                self.mass_Zderiv[i,self.quad[a[j],:]] = self.mass_Zderiv[i,self.quad[a[j],:]] - self.volume_knock_down*self.fuel_density*A_Z*(Z1+Z2+Z3+Z4)/4
                self.mass_Zderiv[i,self.quad[a[j],:]] = self.mass_Zderiv[i,self.quad[a[j],:]] - self.volume_knock_down*self.fuel_density*A*np.ones(4)/4
        
    def _span_division(self,Y):
        
        ## compute spanwise subdivisions
        
        y = np.min(Y) + np.linspace(0,self.N_span,self.N_span+1)*(np.max(Y)-np.min(Y))/self.N_span
        
        ## compute deriavtive of spanwise subdivision wrt Y
        
        y_Yderiv = np.zeros([len(y),len(Y)])

        j1 = np.argmin(Y)
        j2 = np.argmax(Y)
        
        y_Yderiv[:,j1] = np.linspace(1.,0.,self.N_span+1)
        y_Yderiv[:,j2] = np.linspace(0.,1.,self.N_span+1)
        
        return y, y_Yderiv
        
    def _element_area(self,X1,X2,X3,X4,Y1,Y2,Y3,Y4,Z1,Z2,Z3,Z4):
        
        v12 = np.array([X2-X1,Y2-Y1,Z2-Z1])
        v13 = np.array([X3-X1,Y3-Y1,Z3-Z1])
        v43 = np.array([X3-X4,Y3-Y4,Z3-Z4])
        v42 = np.array([X2-X4,Y2-Y4,Z2-Z4])

        v1 = np.cross(v12,v13)
        n1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        v2 = np.cross(v43,v42)
        n2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        A = .5*(n1+n2)
        
        v1_X = np.array([[0,Z3-Z2,Y2-Y3],[0,Z1-Z3,Y3-Y1],[0,Z2-Z1,Y1-Y2],[0,0,0]])
        v1_Y = np.array([[Z2-Z3,0,X3-X2],[Z3-Z1,0,X1-X3],[Z1-Z2,0,X2-X1],[0,0,0]])
        v1_Z = np.array([[Y3-Y2,X2-X3,0],[Y1-Y3,X3-X1,0],[Y2-Y1,X1-X2,0],[0,0,0]])
        v2_X = np.array([[0,0,0],[0,Z3-Z4,Y4-Y3],[0,Z4-Z2,Y2-Y4],[0,Z2-Z3,Y3-Y2]])
        v2_Y = np.array([[0,0,0],[Z4-Z3,0,X3-X4],[Z2-Z4,0,X4-X2],[Z3-Z2,0,X2-X3]])
        v2_Z = np.array([[0,0,0],[Y3-Y4,X4-X3,0],[Y4-Y2,X2-X4,0],[Y2-Y3,X3-X2,0]])
                            
        A_X = .5*(v1@v1_X.transpose()/n1 + v2@v2_X.transpose()/n2)
        A_Y = .5*(v1@v1_Y.transpose()/n1 + v2@v2_Y.transpose()/n2)
        A_Z = .5*(v1@v1_Z.transpose()/n1 + v2@v2_Z.transpose()/n2)
                        
        return A, A_X, A_Y, A_Z

## obtain the finite elements which are a member of a given component patch

def _component_elements(patch,prop_ID):

    el = []
    for i in range(0,np.shape(patch)[0]):
        for j in range(0,np.shape(patch)[1]):
            el = np.r_[el,np.squeeze(np.argwhere(prop_ID==patch[i,j]-1))]
        
    el = el.astype(int)
            
    return el                
        
## explicit om component which computes fuel mass and fuelforces

class FuelLoads(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('N_nodes', types=int)
        self.options.declare('elements', types=np.ndarray)
        self.options.declare('prop_ID', types=np.ndarray)
        self.options.declare('patches')
       
        self.gravity = -9.81
        self.reserve_fuel = 7500.
        self.fuel_density = 810.
 
    def setup(self):

        self.add_input('x_s0',np.zeros(self.options['N_nodes']*3))
        self.add_input('fuel_DV',0.0)
        self.add_input('load_factor',0.0)
        self.add_input('load_case_fuel_burned',0.0)        

        self.add_output('fuel_mass',0.0)
        self.add_output('F_fuel',np.zeros(self.options['N_nodes']*6))  

    def compute(self,inputs,outputs):
            
        gravity = self.gravity*inputs['load_factor']
        quad = self.options['elements']
        
        X = inputs['x_s0'][0::3]
        Y = inputs['x_s0'][1::3]
        Z = inputs['x_s0'][2::3]
        
        ## compute available fuel mass
        
        self.fuel = FuelMass(inputs['x_s0'],quad,self.options['prop_ID'],self.options['patches'],self.fuel_density)
        self.fuel.set_CS(self.under_complex_step)
        self.fuel.compute()
                                       
        ## find fuel_fraction: percentage of fuel_mass seen by this load case

        self.fuel_fraction = (inputs['load_case_fuel_burned']*(inputs['fuel_DV']*np.sum(self.fuel.mass) - self.reserve_fuel) + \
                         self.reserve_fuel)/np.sum(self.fuel.mass)

        ## final fuel mass output
        
        outputs['fuel_mass'] = self.fuel_fraction*np.sum(self.fuel.mass)

        ## find rib-spar-skin intersection points

        US = _component_elements(self.options['patches'].upper_skin,self.options['prop_ID'])
        US = np.unique(quad[US].flatten())
        
        LS = _component_elements(self.options['patches'].lower_skin,self.options['prop_ID'])
        LS = np.unique(quad[LS].flatten())
        
        LE = _component_elements(self.options['patches'].le_spar,self.options['prop_ID'])
        LE = np.unique(quad[LE].flatten())
        
        TE = _component_elements(self.options['patches'].te_spar,self.options['prop_ID'])
        TE = np.unique(quad[TE].flatten())
        
        rib = _component_elements(self.options['patches'].rib,self.options['prop_ID'])
        rib = np.unique(quad[rib].flatten())
        
        top_front = np.intersect1d(np.intersect1d(US,LE),rib)
        bot_front = np.intersect1d(np.intersect1d(LS,LE),rib)
        top_rear = np.intersect1d(np.intersect1d(US,TE),rib)
        bot_rear = np.intersect1d(np.intersect1d(LS,TE),rib)
        
        ## find fuel force vector

        outputs['F_fuel'] = np.zeros(self.options['N_nodes']*6)
        
        self.connect = np.zeros([8,len(self.fuel.mass)],dtype=int)
        
        for i in range(0,len(self.fuel.mass)):
            
            ## find 8 connection points
    
            j = np.argwhere(Y[top_front] < (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[-1]
            self.connect[0,i] = top_front[j]
    
            j = np.argwhere(Y[bot_front] < (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[-1]
            self.connect[1,i] = bot_front[j]

            j = np.argwhere(Y[bot_rear] < (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[-1]
            self.connect[2,i] = bot_rear[j]
    
            j = np.argwhere(Y[top_rear] < (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[-1]
            self.connect[3,i] = top_rear[j]

            j = np.argwhere(Y[top_front] > (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[0]
            self.connect[4,i] = top_front[j]
    
            j = np.argwhere(Y[bot_front] > (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[0]
            self.connect[5,i] = bot_front[j]

            j = np.argwhere(Y[bot_rear] > (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[0]
            self.connect[6,i] = bot_rear[j]
    
            j = np.argwhere(Y[top_rear] > (self.fuel.y[i+1]+self.fuel.y[i])/2)
            j = j[0]
            self.connect[7,i] = top_rear[j]
    
            ## RBE terms
            
            T,_,_,_,_,_,_ = self._RBE3(self.fuel.x[i],(self.fuel.y[i+1]+self.fuel.y[i])/2,self.fuel.z[i],\
                           X[self.connect[:,i]],Y[self.connect[:,i]],Z[self.connect[:,i]])
            
            ## final force
    
            f = gravity*self.fuel_fraction*T.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0])
            
            outputs['F_fuel'][self.connect[:,i]*6+0] = outputs['F_fuel'][self.connect[:,i]*6+0]  + f[0::6]
            outputs['F_fuel'][self.connect[:,i]*6+1] = outputs['F_fuel'][self.connect[:,i]*6+1]  + f[1::6]
            outputs['F_fuel'][self.connect[:,i]*6+2] = outputs['F_fuel'][self.connect[:,i]*6+2]  + f[2::6]
         
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode): 

        gravity = self.gravity*inputs['load_factor']

        X = inputs['x_s0'][0::3]
        Y = inputs['x_s0'][1::3]
        Z = inputs['x_s0'][2::3]
        
        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            
            ## derivative of fuel_fraction: percentage of fuel_mass seen by this load case
        
            fuel_fraction_X = inputs['load_case_fuel_burned']*inputs['fuel_DV']*np.sum(self.fuel.mass_Xderiv,axis=0)/np.sum(self.fuel.mass) - \
                             (inputs['load_case_fuel_burned']*(inputs['fuel_DV']*np.sum(self.fuel.mass) - self.reserve_fuel) + \
                              self.reserve_fuel)*np.sum(self.fuel.mass_Xderiv,axis=0)/np.sum(self.fuel.mass)/np.sum(self.fuel.mass)

            fuel_fraction_Y = inputs['load_case_fuel_burned']*inputs['fuel_DV']*np.sum(self.fuel.mass_Yderiv,axis=0)/np.sum(self.fuel.mass) - \
                             (inputs['load_case_fuel_burned']*(inputs['fuel_DV']*np.sum(self.fuel.mass) - self.reserve_fuel) + \
                              self.reserve_fuel)*np.sum(self.fuel.mass_Yderiv,axis=0)/np.sum(self.fuel.mass)/np.sum(self.fuel.mass)

            fuel_fraction_Z = inputs['load_case_fuel_burned']*inputs['fuel_DV']*np.sum(self.fuel.mass_Zderiv,axis=0)/np.sum(self.fuel.mass) - \
                             (inputs['load_case_fuel_burned']*(inputs['fuel_DV']*np.sum(self.fuel.mass) - self.reserve_fuel) + \
                              self.reserve_fuel)*np.sum(self.fuel.mass_Zderiv,axis=0)/np.sum(self.fuel.mass)/np.sum(self.fuel.mass)

            fuel_fraction_fuel_DV = inputs['load_case_fuel_burned']
        
            ## derivative of fuel mass output
            
            if 'fuel_mass' in d_outputs:
                if 'x_s0' in d_inputs:
                        
                    d_inputs['x_s0'][0::3] += d_outputs['fuel_mass']*(fuel_fraction_X*np.sum(self.fuel.mass) + self.fuel_fraction*np.sum(self.fuel.mass_Xderiv,axis=0))
                    d_inputs['x_s0'][1::3] += d_outputs['fuel_mass']*(fuel_fraction_Y*np.sum(self.fuel.mass) + self.fuel_fraction*np.sum(self.fuel.mass_Yderiv,axis=0))
                    d_inputs['x_s0'][2::3] += d_outputs['fuel_mass']*(fuel_fraction_Z*np.sum(self.fuel.mass) + self.fuel_fraction*np.sum(self.fuel.mass_Zderiv,axis=0))
                    
                if 'fuel_DV' in d_inputs:
                        
                    d_inputs['fuel_DV'] += d_outputs['fuel_mass']*fuel_fraction_fuel_DV*np.sum(self.fuel.mass)

            ## find derivative of the fuel force vector
            
            for i in range(0,len(self.fuel.mass)):
                
                T,T_x,T_y,T_z,T_x_in,T_y_in,T_z_in = self._RBE3(\
                                self.fuel.x[i],(self.fuel.y[i+1]+self.fuel.y[i])/2,self.fuel.z[i],\
                                X[self.connect[:,i]],Y[self.connect[:,i]],Z[self.connect[:,i]])
                
                ## final force derivatives wrt x coordinates
                
                f_X = gravity*np.outer(T.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0]),fuel_fraction_X) + \
                      gravity*self.fuel_fraction*np.outer(T_x.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0]),self.fuel.x_Xderiv[i,:]) + \
                      gravity*self.fuel_fraction*np.outer(T.transpose()@np.array([0,0,1.,0,0,0]),self.fuel.mass_Xderiv[i,:])
                
                for j in range(0,8):
                    f_X[:,self.connect[j,i]] = f_X[:,self.connect[j,i]] + gravity*self.fuel_fraction*T_x_in[:,:,j].transpose()@np.array([0,0,self.fuel.mass[i],0,0,0])
    
                ## final force derivatives wrt y coordinates
    
                f_Y = gravity*np.outer(T.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0]),fuel_fraction_Y) + \
                      gravity*self.fuel_fraction*np.outer(T_y.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0]),(self.fuel.y_Yderiv[i,:]+self.fuel.y_Yderiv[i+1,:])/2) + \
                      gravity*self.fuel_fraction*np.outer(T.transpose()@np.array([0,0,1.,0,0,0]),self.fuel.mass_Yderiv[i,:])
                      
                for j in range(0,8):
                    f_Y[:,self.connect[j,i]] = f_Y[:,self.connect[j,i]] + gravity*self.fuel_fraction*T_y_in[:,:,j].transpose()@np.array([0,0,self.fuel.mass[i],0,0,0])
    
                ## final force derivatives wrt z coordinates
                
                f_Z = gravity*np.outer(T.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0]),fuel_fraction_Z) + \
                      gravity*self.fuel_fraction*np.outer(T_z.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0]),self.fuel.z_Zderiv[i,:]) + \
                      gravity*self.fuel_fraction*np.outer(T.transpose()@np.array([0,0,1.,0,0,0]),self.fuel.mass_Zderiv[i,:])
                
                for j in range(0,8):
                    f_Z[:,self.connect[j,i]] = f_Z[:,self.connect[j,i]] + gravity*self.fuel_fraction*T_z_in[:,:,j].transpose()@np.array([0,0,self.fuel.mass[i],0,0,0])
                    
                ## final force derivatives wrt fuel_DV
                
                f_fuel_DV = gravity*fuel_fraction_fuel_DV*T.transpose()@np.array([0,0,self.fuel.mass[i],0,0,0])
          
                ## assemble
                
                if 'F_fuel' in d_outputs:
                    if 'x_s0' in d_inputs:
                        
                        d_inputs['x_s0'][0::3] += f_X[0::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+0]
                        d_inputs['x_s0'][0::3] += f_X[1::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+1]
                        d_inputs['x_s0'][0::3] += f_X[2::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+2]
                        
                        d_inputs['x_s0'][1::3] += f_Y[0::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+0]
                        d_inputs['x_s0'][1::3] += f_Y[1::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+1]
                        d_inputs['x_s0'][1::3] += f_Y[2::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+2]
                        
                        d_inputs['x_s0'][2::3] += f_Z[0::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+0]
                        d_inputs['x_s0'][2::3] += f_Z[1::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+1]
                        d_inputs['x_s0'][2::3] += f_Z[2::6,:].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+2]   
                        
                    if 'fuel_DV' in d_inputs:
                        
                        d_inputs['fuel_DV'] += f_fuel_DV[0::6].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+0]
                        d_inputs['fuel_DV'] += f_fuel_DV[1::6].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+0]
                        d_inputs['fuel_DV'] += f_fuel_DV[2::6].transpose()@d_outputs['F_fuel'][self.connect[:,i]*6+0]
                        
                
    def _RBE3(self,x,y,z,x_in,y_in,z_in):
        
        W = np.kron(np.eye(len(x_in)),np.diag([1,1,1,0,0,0]))
        
        if self.under_complex_step is True:
            S = np.zeros([6*len(x_in),6],dtype=complex)
        else:
            S = np.zeros([6*len(x_in),6])
        
        for j in range(0,len(x_in)):
            Lx = x_in[j]-x
            Ly = y_in[j]-y
            Lz = z_in[j]-z
            S[j*6:j*6+6,:] = np.array([[1,0,0,0,Lz,-Ly],[0,1,0,-Lz,0,Lx],[0,0,1,Ly,-Lx,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

        T = np.linalg.solve(S.transpose()@W@S,S.transpose()@W)
        
        ## compute derivative of T wrt x
        
        S_deriv = S*0
        for j in range(0,len(x_in)):
            Lx = -1.
            Ly = 0.
            Lz = 0.  
            S_deriv[j*6:j*6+6,:] = np.array([[0,0,0,0,Lz,-Ly],[0,0,0,-Lz,0,Lx],[0,0,0,Ly,-Lx,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
        
        T_x = np.linalg.solve(S.transpose()@W@S,S_deriv.transpose()@W-(S_deriv.transpose()@W@S+S.transpose()@W@S_deriv)@T)
        
        ## compute derivative of T wrt y
        
        S_deriv = S*0
        for j in range(0,len(x_in)):
            Lx = 0.
            Ly = -1.
            Lz = 0.  
            S_deriv[j*6:j*6+6,:] = np.array([[0,0,0,0,Lz,-Ly],[0,0,0,-Lz,0,Lx],[0,0,0,Ly,-Lx,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
        
        T_y = np.linalg.solve(S.transpose()@W@S,S_deriv.transpose()@W-(S_deriv.transpose()@W@S+S.transpose()@W@S_deriv)@T)
        
        ## compute derivative of T wrt z
        
        S_deriv = S*0
        for j in range(0,len(x_in)):
            Lx = 0.
            Ly = 0.
            Lz = -1.  
            S_deriv[j*6:j*6+6,:] = np.array([[0,0,0,0,Lz,-Ly],[0,0,0,-Lz,0,Lx],[0,0,0,Ly,-Lx,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
        
        T_z = np.linalg.solve(S.transpose()@W@S,S_deriv.transpose()@W-(S_deriv.transpose()@W@S+S.transpose()@W@S_deriv)@T)
                
        ## compute derivative of T wrt x_in
        
        if self.under_complex_step is True:
            T_x_in = np.zeros([6,6*len(x_in),len(x_in)],dtype=complex)
        else:
            T_x_in = np.zeros([6,6*len(x_in),len(x_in)])   
                
        for j in range(0,len(x_in)):
            Lx = 1.
            Ly = 0.
            Lz = 0.
            S_deriv = S*0
            S_deriv[j*6:j*6+6,:] = np.array([[0,0,0,0,Lz,-Ly],[0,0,0,-Lz,0,Lx],[0,0,0,Ly,-Lx,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
            T_x_in[:,:,j] = np.linalg.solve(S.transpose()@W@S,S_deriv.transpose()@W-(S_deriv.transpose()@W@S+S.transpose()@W@S_deriv)@T)
        
        ## compute derivative of T wrt y_in
        
        if self.under_complex_step is True:
            T_y_in = np.zeros([6,6*len(x_in),len(x_in)],dtype=complex)
        else:
            T_y_in = np.zeros([6,6*len(x_in),len(x_in)])   
            
        for j in range(0,len(x_in)):
            Lx = 0.
            Ly = 1.
            Lz = 0.
            S_deriv = S*0
            S_deriv[j*6:j*6+6,:] = np.array([[0,0,0,0,Lz,-Ly],[0,0,0,-Lz,0,Lx],[0,0,0,Ly,-Lx,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
            T_y_in[:,:,j] = np.linalg.solve(S.transpose()@W@S,S_deriv.transpose()@W-(S_deriv.transpose()@W@S+S.transpose()@W@S_deriv)@T)
                
        ## compute derivative of T wrt z_in
        
        if self.under_complex_step is True:
            T_z_in = np.zeros([6,6*len(x_in),len(x_in)],dtype=complex)
        else:
            T_z_in = np.zeros([6,6*len(x_in),len(x_in)])   
            
        for j in range(0,len(x_in)):
            Lx = 0.
            Ly = 0.
            Lz = 1.
            S_deriv = S*0
            S_deriv[j*6:j*6+6,:] = np.array([[0,0,0,0,Lz,-Ly],[0,0,0,-Lz,0,Lx],[0,0,0,Ly,-Lx,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
            T_z_in[:,:,j] = np.linalg.solve(S.transpose()@W@S,S_deriv.transpose()@W-(S_deriv.transpose()@W@S+S.transpose()@W@S_deriv)@T)
                  
        return T, T_x, T_y, T_z, T_x_in, T_y_in, T_z_in

