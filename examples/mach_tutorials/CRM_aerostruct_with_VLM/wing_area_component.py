import numpy as np
import openmdao.api as om
   
class WingArea:
    
    def __init__(self,nodes,quads):
        
        self.nodes = nodes
        self.quad = quads
    
    def compute(self):
        
        ## unpack coordinates
        
        X = self.nodes[0::3]
        Y = self.nodes[1::3]
        Z = self.nodes[2::3]
        
        ## compute planform area by looping over all surface elements, but leaving the z-coordinate as 0
        
        self.A = 0.*X[0]
        self.A_X = np.zeros(len(X))
        self.A_Y = np.zeros(len(Y))
        
        for i in range(0,len(self.quad)):
            
            X1 = X[self.quad[i,0]-1]; X2 = X[self.quad[i,1]-1]; X3 = X[self.quad[i,2]-1]; X4 = X[self.quad[i,3]-1];
            Y1 = Y[self.quad[i,0]-1]; Y2 = Y[self.quad[i,1]-1]; Y3 = Y[self.quad[i,2]-1]; Y4 = Y[self.quad[i,3]-1];

            v12 = np.array([X2-X1,Y2-Y1,0.])
            v13 = np.array([X3-X1,Y3-Y1,0.])
            v14 = np.array([X4-X1,Y4-Y1,0.])

            v1 = np.cross(v12,v13)
            v2 = np.cross(v13,v14)
            
            n1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            n2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
            
            self.A = self.A + .5*(n1+n2)
            
            v1_X = np.array([[0,0,Y2-Y3],[0,0,Y3-Y1],[0,0,Y1-Y2],[0,0,0]])
            v1_Y = np.array([[0,0,X3-X2],[0,0,X1-X3],[0,0,X2-X1],[0,0,0]])
            v2_X = np.array([[0,0,Y3-Y4],[0,0,0],[0,0,Y4-Y1],[0,0,Y1-Y3]])
            v2_Y = np.array([[0,0,X4-X3],[0,0,0],[0,0,X1-X4],[0,0,X3-X1]])
            
            self.A_X[self.quad[i,:]-1] = self.A_X[self.quad[i,:]-1] + .5*(v1@v1_X.transpose()/n1 + v2@v2_X.transpose()/n2)
            self.A_Y[self.quad[i,:]-1] = self.A_Y[self.quad[i,:]-1] + .5*(v1@v1_Y.transpose()/n1 + v2@v2_Y.transpose()/n2)
            
        ## divide by 2, b/c you just computed the area of the upper and lower surfaces
        
        self.A = self.A/2
        self.A_X = self.A_X/2
        self.A_Y = self.A_Y/2
        

## OM component to compute wing area
        
class WingAreaComponent(om.ExplicitComponent):
    
    def initialize(self): 
        
        self.options.declare('N_nodes', types=int)
        self.options.declare('quad', types=np.ndarray)
        
    def setup(self):

        self.add_input('x',np.zeros(self.options['N_nodes']*3))
        self.add_output('area',0.0)
        
    def compute(self,inputs,outputs):
        
        self.Area = WingArea(inputs['x'], self.options['quad'])
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
