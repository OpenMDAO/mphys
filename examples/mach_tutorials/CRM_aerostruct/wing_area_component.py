import numpy as np
import openmdao.api as om
   
class WingArea:
    
    def __init__(self,nodes,connect):
        
        self.nodes = nodes
        self.connect = connect
    
    def compute(self):
        
        ## unpack coordinates
        
        X = self.nodes[0::3]
        Y = self.nodes[1::3]
        Z = self.nodes[2::3]
        
        ## compute planform area by looping over all surface elements, but leaving the z-coordinate as 0
        
        self.A = 0.*X[0]
        self.A_X = np.zeros(len(X))
        self.A_Y = np.zeros(len(Y))
        
        for i in range(0,len(self.connect)):
            
            X1 = X[self.connect[i,0]-1]; X2 = X[self.connect[i,1]-1]; X3 = X[self.connect[i,2]-1]; 
            Y1 = Y[self.connect[i,0]-1]; Y2 = Y[self.connect[i,1]-1]; Y3 = Y[self.connect[i,2]-1]; 

            v12 = np.array([X2-X1,Y2-Y1,0.])
            v13 = np.array([X3-X1,Y3-Y1,0.])

            v1 = np.cross(v12,v13)
            n1 = v1[2]
            
            self.A = self.A + .5*n1*np.sign(np.real(n1))
            
            A_n1 = np.sign(np.real(n1))/2
            n1_v1 = np.array([0,0,1.])

            v1_X = np.array([[0,0,Y2-Y3],[0,0,Y3-Y1],[0,0,Y1-Y2]])
            v1_Y = np.array([[0,0,X3-X2],[0,0,X1-X3],[0,0,X2-X1]])
            
            self.A_X[self.connect[i,:]-1] = self.A_X[self.connect[i,:]-1] + A_n1*n1_v1@v1_X.transpose()
            self.A_Y[self.connect[i,:]-1] = self.A_Y[self.connect[i,:]-1] + A_n1*n1_v1@v1_Y.transpose()
            
        ## divide by 2, b/c you just computed the area of the upper and lower surfaces
        
        self.A = self.A/2
        self.A_X = self.A_X/2
        self.A_Y = self.A_Y/2
        

## OM component to compute wing area
        
class WingAreaComponent(om.ExplicitComponent):
    
    def initialize(self): 
        
        self.options.declare('N_nodes', types=int)
        self.options.declare('connect', types=np.ndarray)
        
    def setup(self):

        self.add_input('x',np.zeros(self.options['N_nodes']*3))
        self.add_output('area',0.0)
        
    def compute(self,inputs,outputs):
        
        self.Area = WingArea(inputs['x'], self.options['connect'])
        self.Area.compute()
        
        outputs['area'] = self.Area.A
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode): 
        
        if mode == 'fwd':
            pass
        
        if mode == 'rev':
            
            if 'area' in d_outputs:
                if 'x' in d_inputs:
                    d_inputs['x'][0::3] += self.Area.A_X*d_outputs['area']
                    d_inputs['x'][1::3] += self.Area.A_Y*d_outputs['area']

