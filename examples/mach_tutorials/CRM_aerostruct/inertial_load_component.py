import numpy as np
import openmdao.api as om

class InertialLoads(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('N_nodes', types=int)
        self.options.declare('elements', types=np.ndarray)
        self.options.declare('prop_ID', types=np.ndarray)
        self.options.declare('n_dvs', types=int)
        self.options.declare('rho', types=float)
        self.options.declare('gravity', types=float) 
       
    def setup(self):

        self.add_input('x_s0',np.zeros(self.options['N_nodes']*3))
        self.add_input('dv_struct',np.zeros(self.options['n_dvs']))
        self.add_input('load_factor',0.0)
 
        self.add_output('F_inertial',np.zeros(self.options['N_nodes']*6))

    def compute(self,inputs,outputs):
         
        gravity = -self.options['gravity']*inputs['load_factor']
        rho = self.options['rho']
        quad = self.options['elements']
                
        X = inputs['x_s0'][0::3]
        Y = inputs['x_s0'][1::3]
        Z = inputs['x_s0'][2::3]

        # loop over elements
        
        outputs['F_inertial'] = np.zeros(self.options['N_nodes']*6)
        
        for i in range(0,len(quad)):
            
            # compute element area
            
            X1 = X[quad[i,0]]; X2 = X[quad[i,1]]; X3 = X[quad[i,2]]; X4 = X[quad[i,3]];
            Y1 = Y[quad[i,0]]; Y2 = Y[quad[i,1]]; Y3 = Y[quad[i,2]]; Y4 = Y[quad[i,3]];
            Z1 = Z[quad[i,0]]; Z2 = Z[quad[i,1]]; Z3 = Z[quad[i,2]]; Z4 = Z[quad[i,3]];
    
            v12 = np.array([X2-X1,Y2-Y1,Z2-Z1])
            v13 = np.array([X3-X1,Y3-Y1,Z3-Z1])
            v43 = np.array([X3-X4,Y3-Y4,Z3-Z4])
            v42 = np.array([X2-X4,Y2-Y4,Z2-Z4])

            v1 = np.cross(v12,v13)
            n1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            v2 = np.cross(v43,v42)
            n2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
            A = .5*(n1+n2)
            
            ## element thickness
            
            t = inputs['dv_struct'][self.options['prop_ID'][i]]
            
            # force
    
            outputs['F_inertial'][quad[i,:]*6+2] = outputs['F_inertial'][quad[i,:]*6+2] + A*gravity*rho*t/4.
         
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):   

        gravity = -self.options['gravity']*inputs['load_factor']
        rho = self.options['rho']
        quad = self.options['elements']
        
        X = inputs['x_s0'][0::3]
        Y = inputs['x_s0'][1::3]
        Z = inputs['x_s0'][2::3]
        
        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            
            # loop over elements
        
            for i in range(0,len(quad)):
        
                # compute element area
            
                X1 = X[quad[i,0]]; X2 = X[quad[i,1]]; X3 = X[quad[i,2]]; X4 = X[quad[i,3]];
                Y1 = Y[quad[i,0]]; Y2 = Y[quad[i,1]]; Y3 = Y[quad[i,2]]; Y4 = Y[quad[i,3]];
                Z1 = Z[quad[i,0]]; Z2 = Z[quad[i,1]]; Z3 = Z[quad[i,2]]; Z4 = Z[quad[i,3]];
    
                v12 = np.array([X2-X1,Y2-Y1,Z2-Z1])
                v13 = np.array([X3-X1,Y3-Y1,Z3-Z1])
                v43 = np.array([X3-X4,Y3-Y4,Z3-Z4])
                v42 = np.array([X2-X4,Y2-Y4,Z2-Z4])

                v1 = np.cross(v12,v13)
                n1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
                v2 = np.cross(v43,v42)
                n2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
                A = .5*(n1+n2)            
                
                # element thickness
            
                t = inputs['dv_struct'][self.options['prop_ID'][i]]
                
                # geometry derivatives
                
                if 'F_inertial' in d_outputs:
                    if 'x_s0' in d_inputs:
                        
                        v1_X = np.array([[0,Z3-Z2,Y2-Y3],[0,Z1-Z3,Y3-Y1],[0,Z2-Z1,Y1-Y2],[0,0,0]])
                        v1_Y = np.array([[Z2-Z3,0,X3-X2],[Z3-Z1,0,X1-X3],[Z1-Z2,0,X2-X1],[0,0,0]])
                        v1_Z = np.array([[Y3-Y2,X2-X3,0],[Y1-Y3,X3-X1,0],[Y2-Y1,X1-X2,0],[0,0,0]])
                        v2_X = np.array([[0,0,0],[0,Z3-Z4,Y4-Y3],[0,Z4-Z2,Y2-Y4],[0,Z2-Z3,Y3-Y2]])
                        v2_Y = np.array([[0,0,0],[Z4-Z3,0,X3-X4],[Z2-Z4,0,X4-X2],[Z3-Z2,0,X2-X3]])
                        v2_Z = np.array([[0,0,0],[Y3-Y4,X4-X3,0],[Y4-Y2,X2-X4,0],[Y2-Y3,X3-X2,0]])
                            
                        A_X = .5*(v1@v1_X.transpose()/n1 + v2@v2_X.transpose()/n2)
                        A_Y = .5*(v1@v1_Y.transpose()/n1 + v2@v2_Y.transpose()/n2)
                        A_Z = .5*(v1@v1_Z.transpose()/n1 + v2@v2_Z.transpose()/n2)
                        
                        d_inputs['x_s0'][quad[i,:]*3+0] += np.tile(A_X,(4,1)).transpose()@d_outputs['F_inertial'][quad[i,:]*6+2]*gravity*rho*t/4
                        d_inputs['x_s0'][quad[i,:]*3+1] += np.tile(A_Y,(4,1)).transpose()@d_outputs['F_inertial'][quad[i,:]*6+2]*gravity*rho*t/4
                        d_inputs['x_s0'][quad[i,:]*3+2] += np.tile(A_Z,(4,1)).transpose()@d_outputs['F_inertial'][quad[i,:]*6+2]*gravity*rho*t/4
                        
                # sizing derivatives
                
                if 'F_inertial' in d_outputs:
                    if 'dv_struct' in d_inputs:
                        
                        d_inputs['dv_struct'][self.options['prop_ID'][i]] += d_outputs['F_inertial'][quad[i,0]*6+2]*A*gravity*rho/4
                        d_inputs['dv_struct'][self.options['prop_ID'][i]] += d_outputs['F_inertial'][quad[i,1]*6+2]*A*gravity*rho/4
                        d_inputs['dv_struct'][self.options['prop_ID'][i]] += d_outputs['F_inertial'][quad[i,2]*6+2]*A*gravity*rho/4
                        d_inputs['dv_struct'][self.options['prop_ID'][i]] += d_outputs['F_inertial'][quad[i,3]*6+2]*A*gravity*rho/4
                        
