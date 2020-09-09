import numpy as np

import openmdao.api as om

class WingGeometry(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('xs', types=np.ndarray)
        self.options.declare('xa', types=np.ndarray)
        self.options.declare('y_knot', types=np.ndarray)
        self.options.declare('LE_knot', types=np.ndarray)
        self.options.declare('TE_knot', types=np.ndarray)

    def setup(self):
        
        self.y_knot = self.options['y_knot']
        self.LE_knot = self.options['LE_knot']
        self.TE_knot = self.options['TE_knot']
        self.N_knots = len(self.y_knot)
        
        self.add_input('root_chord_delta')
        self.add_input('tip_chord_delta')
        self.add_input('tip_sweep_delta')
        self.add_input('span_delta')
        self.add_input('wing_thickness_delta',shape=self.N_knots-1)
        self.add_input('wing_twist_delta',shape=(self.N_knots-1))
        
        self.add_output('x_s0_mesh',shape=len(self.options['xs']))
        self.add_output('x_a0_mesh',shape=len(self.options['xa']))
        
        self.declare_partials('x_s0_mesh','root_chord_delta')
        self.declare_partials('x_s0_mesh','tip_chord_delta')
        self.declare_partials('x_s0_mesh','tip_sweep_delta')
        self.declare_partials('x_s0_mesh','span_delta')
        self.declare_partials('x_s0_mesh','wing_thickness_delta')
        self.declare_partials('x_s0_mesh','wing_twist_delta')
        self.declare_partials('x_a0_mesh','root_chord_delta')
        self.declare_partials('x_a0_mesh','tip_chord_delta')
        self.declare_partials('x_a0_mesh','tip_sweep_delta')
        self.declare_partials('x_a0_mesh','span_delta')
        self.declare_partials('x_a0_mesh','wing_thickness_delta')
        self.declare_partials('x_a0_mesh','wing_twist_delta')

        self.set_aero_grids(self.options['xa'])
        self.set_structure_grids(self.options['xs'])
        
    def compute(self,inputs,outputs):
        
        self.set_root_chord_DV(inputs['root_chord_delta'])
        self.set_tip_chord_DV(inputs['tip_chord_delta'])
        self.set_tip_sweep_DV(inputs['tip_sweep_delta'])
        self.set_span_DV(inputs['span_delta'])
        self.set_thickness_DVs(inputs['wing_thickness_delta'])
        self.set_twist_DVs(inputs['wing_twist_delta'])

        self.complex_step = self.under_complex_step
        
        self.morph_wing()
        
        outputs['x_s0_mesh'] = self.xs_new
        outputs['x_a0_mesh'] = self.xa_new
    
    def compute_partials(self,inputs,partials):
        
        self.compute_derivatives()
        
        partials['x_s0_mesh','root_chord_delta'] = self.xs_root_chord_DV
        partials['x_s0_mesh','tip_chord_delta'] = self.xs_tip_chord_DV
        partials['x_s0_mesh','tip_sweep_delta'] = self.xs_tip_sweep_DV
        partials['x_s0_mesh','span_delta'] = self.xs_span_DV
        partials['x_s0_mesh','wing_thickness_delta'] = self.xs_thickness_DV
        partials['x_s0_mesh','wing_twist_delta'] = self.xs_twist_DV

        partials['x_a0_mesh','root_chord_delta'] = self.xa_root_chord_DV
        partials['x_a0_mesh','tip_chord_delta'] = self.xa_tip_chord_DV
        partials['x_a0_mesh','tip_sweep_delta'] = self.xa_tip_sweep_DV
        partials['x_a0_mesh','span_delta'] = self.xa_span_DV
        partials['x_a0_mesh','wing_thickness_delta'] = self.xa_thickness_DV
        partials['x_a0_mesh','wing_twist_delta'] = self.xa_twist_DV
        
    def set_aero_grids(self, xa):
        """
        Sets the aero grids

        """
        self.xa = xa 
        
        self.Xa = self.xa[0::3]
        self.Ya = self.xa[1::3]
        self.Za = self.xa[2::3]
               
        
    def set_structure_grids(self, xs):
        """
        Sets the structure grids

        """
        self.xs = xs 
        
        self.Xs = self.xs[0::3]
        self.Ys = self.xs[1::3]
        self.Zs = self.xs[2::3]
               
        
    def set_root_chord_DV(self, root_chord_DV):
        """
        Sets the root chord morphing design variable

        """
        self.root_chord_DV = root_chord_DV 
             
        
    def set_tip_chord_DV(self, tip_chord_DV):
        """
        Sets the tip chord morphing design variable

        """
        self.tip_chord_DV = tip_chord_DV 
               
        
    def set_tip_sweep_DV(self, tip_sweep_DV):
        """
        Sets the sweep morphing design variable

        """
        self.tip_sweep_DV = tip_sweep_DV 
               
        
    def set_span_DV(self, span_DV):
        """
        Sets the span morphing design variable

        """
        self.span_DV = span_DV         
        
        
    def set_thickness_DVs(self, thickness_DV):
        """
        Sets the array of thickness morphing design variables

        """
        self.thickness_DV = thickness_DV 
        
        
    def set_twist_DVs(self, twist_DV):
        """
        Sets the array of twist morphing design variables

        """
        self.twist_DV = twist_DV
        
        
    def morph_wing(self):
        """
        Morphs the aero and structural grids using the DVs

        """
        
        ## find z-coordinates of a bounding box which encloses the wing

        self._define_bounding_box()
        
        ## morph the aero mesh
        
        Xa_new, Ya_new, Za_new = self._morphing_fun(self.Xa,self.Ya,self.Za)
        
        self.xa_new = np.c_[Xa_new,Ya_new,Za_new].flatten(order='C')

        ## morph the structure mesh
        
        Xs_new, Ys_new, Zs_new = self._morphing_fun(self.Xs,self.Ys,self.Zs)
        
        self.xs_new = np.c_[Xs_new,Ys_new,Zs_new].flatten(order='C')
        
    
    def compute_derivatives(self):
        """
        Computes derivatives of the morphed grids with respect to DVs, via complex step

        """        

        CS = 1E-40
        self.complex_step = True
        
        ## root_chord_DV derivatives
        
        self.root_chord_DV = self.root_chord_DV + 1j*CS
        Xa_root_chord_deriv, Ya_root_chord_deriv, Za_root_chord_deriv = self._morphing_fun(self.Xa,self.Ya,self.Za)
        Xs_root_chord_deriv, Ys_root_chord_deriv, Zs_root_chord_deriv = self._morphing_fun(self.Xs,self.Ys,self.Zs)
        self.root_chord_DV = self.root_chord_DV - 1j*CS

        Xs_root_chord_deriv = np.imag(Xs_root_chord_deriv)/CS
        Ys_root_chord_deriv = np.imag(Ys_root_chord_deriv)/CS
        Zs_root_chord_deriv = np.imag(Zs_root_chord_deriv)/CS
        self.xs_root_chord_DV = np.c_[Xs_root_chord_deriv,Ys_root_chord_deriv,Zs_root_chord_deriv].flatten(order='C')
        
        Xa_root_chord_deriv = np.imag(Xa_root_chord_deriv)/CS
        Ya_root_chord_deriv = np.imag(Ya_root_chord_deriv)/CS
        Za_root_chord_deriv = np.imag(Za_root_chord_deriv)/CS
        self.xa_root_chord_DV = np.c_[Xa_root_chord_deriv,Ya_root_chord_deriv,Za_root_chord_deriv].flatten(order='C')
        
        ## tip_chord_DV derivatives
        
        self.tip_chord_DV = self.tip_chord_DV + 1j*CS
        Xa_tip_chord_deriv, Ya_tip_chord_deriv, Za_tip_chord_deriv = self._morphing_fun(self.Xa,self.Ya,self.Za)
        Xs_tip_chord_deriv, Ys_tip_chord_deriv, Zs_tip_chord_deriv = self._morphing_fun(self.Xs,self.Ys,self.Zs)
        self.tip_chord_DV = self.tip_chord_DV - 1j*CS

        Xs_tip_chord_deriv = np.imag(Xs_tip_chord_deriv)/CS
        Ys_tip_chord_deriv = np.imag(Ys_tip_chord_deriv)/CS
        Zs_tip_chord_deriv = np.imag(Zs_tip_chord_deriv)/CS
        self.xs_tip_chord_DV = np.c_[Xs_tip_chord_deriv,Ys_tip_chord_deriv,Zs_tip_chord_deriv].flatten(order='C')
        
        Xa_tip_chord_deriv = np.imag(Xa_tip_chord_deriv)/CS
        Ya_tip_chord_deriv = np.imag(Ya_tip_chord_deriv)/CS
        Za_tip_chord_deriv = np.imag(Za_tip_chord_deriv)/CS
        self.xa_tip_chord_DV = np.c_[Xa_tip_chord_deriv,Ya_tip_chord_deriv,Za_tip_chord_deriv].flatten(order='C')
        
        ## tip_sweep_DV derivatives
        
        self.tip_sweep_DV = self.tip_sweep_DV + 1j*CS
        Xa_tip_sweep_deriv, Ya_tip_sweep_deriv, Za_tip_sweep_deriv = self._morphing_fun(self.Xa,self.Ya,self.Za)
        Xs_tip_sweep_deriv, Ys_tip_sweep_deriv, Zs_tip_sweep_deriv = self._morphing_fun(self.Xs,self.Ys,self.Zs)
        self.tip_sweep_DV = self.tip_sweep_DV - 1j*CS

        Xs_tip_sweep_deriv = np.imag(Xs_tip_sweep_deriv)/CS
        Ys_tip_sweep_deriv = np.imag(Ys_tip_sweep_deriv)/CS
        Zs_tip_sweep_deriv = np.imag(Zs_tip_sweep_deriv)/CS
        self.xs_tip_sweep_DV = np.c_[Xs_tip_sweep_deriv,Ys_tip_sweep_deriv,Zs_tip_sweep_deriv].flatten(order='C')
        
        Xa_tip_sweep_deriv = np.imag(Xa_tip_sweep_deriv)/CS
        Ya_tip_sweep_deriv = np.imag(Ya_tip_sweep_deriv)/CS
        Za_tip_sweep_deriv = np.imag(Za_tip_sweep_deriv)/CS
        self.xa_tip_sweep_DV = np.c_[Xa_tip_sweep_deriv,Ya_tip_sweep_deriv,Za_tip_sweep_deriv].flatten(order='C')
        
        ## span_DV derivatives
        
        self.span_DV = self.span_DV + 1j*CS
        Xa_span_deriv, Ya_span_deriv, Za_span_deriv = self._morphing_fun(self.Xa,self.Ya,self.Za)
        Xs_span_deriv, Ys_span_deriv, Zs_span_deriv = self._morphing_fun(self.Xs,self.Ys,self.Zs)
        self.span_DV = self.span_DV - 1j*CS

        Xs_span_deriv = np.imag(Xs_span_deriv)/CS
        Ys_span_deriv = np.imag(Ys_span_deriv)/CS
        Zs_span_deriv = np.imag(Zs_span_deriv)/CS
        self.xs_span_DV = np.c_[Xs_span_deriv,Ys_span_deriv,Zs_span_deriv].flatten(order='C')
        
        Xa_span_deriv = np.imag(Xa_span_deriv)/CS
        Ya_span_deriv = np.imag(Ya_span_deriv)/CS
        Za_span_deriv = np.imag(Za_span_deriv)/CS
        self.xa_span_DV = np.c_[Xa_span_deriv,Ya_span_deriv,Za_span_deriv].flatten(order='C')
        
        ## thickness derivatives
        
        self.thickness_DV = self.thickness_DV.astype('complex')
        
        self.xs_thickness_DV = np.zeros([len(self.xs),len(self.thickness_DV)])
        self.xa_thickness_DV = np.zeros([len(self.xa),len(self.thickness_DV)])
        
        for i in range(0,len(self.thickness_DV)):
            self.thickness_DV[i] = self.thickness_DV[i] + 1j*CS
            Xa_CS, Ya_CS, Za_CS = self._morphing_fun(self.Xa,self.Ya,self.Za)
            Xs_CS, Ys_CS, Zs_CS = self._morphing_fun(self.Xs,self.Ys,self.Zs)
            self.thickness_DV[i] = self.thickness_DV[i] - 1j*CS
    
            self.xs_thickness_DV[0::3,i] = np.imag(Xs_CS)/CS
            self.xs_thickness_DV[1::3,i] = np.imag(Ys_CS)/CS
            self.xs_thickness_DV[2::3,i] = np.imag(Zs_CS)/CS
            
            self.xa_thickness_DV[0::3,i] = np.imag(Xa_CS)/CS
            self.xa_thickness_DV[1::3,i] = np.imag(Ya_CS)/CS
            self.xa_thickness_DV[2::3,i] = np.imag(Za_CS)/CS
            
        self.thickness_DV = np.real(self.thickness_DV)
        
        ## twist derivatives
        
        self.twist_DV = self.twist_DV.astype('complex')
        
        self.xs_twist_DV = np.zeros([len(self.xs),len(self.twist_DV)])
        self.xa_twist_DV = np.zeros([len(self.xa),len(self.twist_DV)])
        
        for i in range(0,len(self.twist_DV)):
            
            self.twist_DV[i] = self.twist_DV[i] + 1j*CS
            Xa_CS, Ya_CS, Za_CS = self._morphing_fun(self.Xa,self.Ya,self.Za)
            Xs_CS, Ys_CS, Zs_CS = self._morphing_fun(self.Xs,self.Ys,self.Zs)
            self.twist_DV[i] = self.twist_DV[i] - 1j*CS
    
            self.xs_twist_DV[0::3,i] = np.imag(Xs_CS)/CS
            self.xs_twist_DV[1::3,i] = np.imag(Ys_CS)/CS
            self.xs_twist_DV[2::3,i] = np.imag(Zs_CS)/CS
            
            self.xa_twist_DV[0::3,i] = np.imag(Xa_CS)/CS
            self.xa_twist_DV[1::3,i] = np.imag(Ya_CS)/CS
            self.xa_twist_DV[2::3,i] = np.imag(Za_CS)/CS
       
        self.twist_DV = np.real(self.twist_DV)
    
    def _define_bounding_box(self):
        """
        Find z-coordinates of a bounding box which encloses the wing
        This is the only part of the code which uses both xa and xs simultaneously
        This process is independent of planform changes

        """   
        
        self.zu_knot = np.zeros(self.N_knots)
        self.zl_knot = np.zeros(self.N_knots)
        
        for i in range(0,len(self.y_knot)):
            fa = np.argwhere(np.abs(self.Ya - self.y_knot[i]) < np.max(self.Ya)/200.)
            fs = np.argwhere(np.abs(self.Ys - self.y_knot[i]) < np.max(self.Ya)/200.)
            
            if fa.size == 0:
                self.zu_knot[i] = np.max(self.Zs[fs])
                self.zl_knot[i] = np.min(self.Zs[fs])
            elif fs.size == 0:
                self.zu_knot[i] = np.max(self.Za[fa])
                self.zl_knot[i] = np.min(self.Za[fa])
            else:
                self.zu_knot[i] = np.max([np.max(self.Za[fa]),np.max(self.Zs[fs])])
                self.zl_knot[i] = np.min([np.min(self.Za[fa]),np.min(self.Zs[fs])])
        
        
    def _morphing_fun(self, X, Y, Z):
        """
        1. Deforms based on span elongation
        2. Deforms based on chord elongation
        3. Deforms based on tip sweep
        4. Deforms based on thickness DVs
        5. Deforms based on twist DVs

        """   
        
        y_knot = self.y_knot
        LE_knot = self.LE_knot
        TE_knot = self.TE_knot
        
        ## deform mesh via span elongation
        
        X,Y = self._span_morph(X,Y,y_knot,LE_knot)
        
        ## deform knots via span elongation
        
        a, b = self._span_morph(np.r_[LE_knot,TE_knot],np.r_[y_knot,y_knot],y_knot,LE_knot)
        LE_knot,TE_knot = np.split(a,2)
        y_knot = np.split(b,2)[0]

        ## deform mesh via chord elongation
        
        X = self._chord_morph(X,Y,y_knot,LE_knot,TE_knot)
        
        ## deform knots via chord elongation

        a = self._chord_morph(np.r_[LE_knot,TE_knot],np.r_[y_knot,y_knot],y_knot,LE_knot,TE_knot)
        LE_knot,TE_knot = np.split(a,2)

        ## deform mesh via tip sweep

        X = self._sweep_morph(X,Y,y_knot)
        
        ## deform knots via tip sweep
        
        a = self._sweep_morph(np.r_[LE_knot,TE_knot],np.r_[y_knot,y_knot],y_knot);
        LE_knot,TE_knot = np.split(a,2)

        ## deform mesh via thickness DVs

        Z = self._thickness_morph(Y,Z,y_knot,self.zl_knot,self.zu_knot)

        ## deform the mesh via twist DVs
        
        Z = self._twist_morph(X, Y, Z, y_knot, LE_knot, TE_knot)
        
        return X, Y, Z
    

    def _span_morph(self, X, Y, y_knot, LE_knot):
        """
        elongate along span direction (doesn't change the chord, keeps the LE sweep constant)
        sweep computed from the 3-thru-N knots

        """   
        
        slope = np.mean((LE_knot[2:] - LE_knot[1:-1])/(y_knot[2:] - y_knot[1:-1]))
        y_shift = self._interp(np.array([y_knot[0],y_knot[1],y_knot[2],y_knot[-1]]),np.array([0.,0.,0.,self.span_DV]),Y)
        
        X = X + y_shift*slope
        Y = Y + y_shift
        
        return X, Y
           
    
    def _chord_morph(self, X, Y, y_knot, LE_knot, TE_knot):
        """
        shift chord by moving the TE only 
        LE doesn't move, and the LE sweep is kept constant

        """   

        eta = (X - self._interp(y_knot,LE_knot,Y))/(self._interp(y_knot,TE_knot,Y) - self._interp(y_knot,LE_knot,Y))
        
        chord = self._interp(y_knot,TE_knot,Y) - self._interp(y_knot,LE_knot,Y)
        del_chord = self._interp(np.array([y_knot[0],y_knot[1],y_knot[-1]]),np.array([self.root_chord_DV,self.root_chord_DV,self.tip_chord_DV]),Y)
        chord_new = chord + del_chord;
        
        X = self._interp(y_knot,LE_knot,Y) + eta*chord_new
        
        return X
        
 
    def _sweep_morph(self, X, Y, y_knot):
        """
        sweep the tip back and forth

        """   
        
        #X = X + self._interp(np.array([y_knot[0],y_knot[1],y_knot[-1]]),np.array([0.,0.,self.tip_sweep_DV]),Y)
        X = X + self._interp(np.array([y_knot[0],y_knot[-1]]),np.array([0.,self.tip_sweep_DV]),Y)

        return X
        
 
    def _thickness_morph(self, Y, Z, y_knot, zl_knot, zu_knot):
        """
        deform mesh via thickness variables

        """  
        
        #thickness = self._RBF_TPS(np.r_[-np.flip(y_knot,0),y_knot[1:]],np.r_[np.flip(self.thickness_DV,0),self.thickness_DV[1:]],Y)
        thickness = self._RBF_TPS(np.r_[-np.flip(y_knot,0),y_knot[1:]],np.r_[np.flip(self.thickness_DV,0),0,self.thickness_DV],Y)
        
        z_eta = (Z - self._interp(y_knot,zl_knot,Y))/(self._interp(y_knot,zu_knot,Y) - self._interp(y_knot,zl_knot,Y))
    
        Z = Z + thickness*z_eta - thickness/2
        
        return Z
        
 
    def _twist_morph(self, X, Y, Z, y_knot, LE_knot, TE_knot):
        """
        deform mesh via thickness variables

        """      
        
        arm = -X + (self._interp(y_knot,LE_knot,Y) + (self._interp(y_knot,TE_knot,Y) - self._interp(y_knot,LE_knot,Y))/2)
             
        twist = self._RBF_TPS(np.r_[-np.flip(y_knot,0),y_knot[1:]],np.r_[np.flip(self.twist_DV,0),0,self.twist_DV],Y)

        Z = Z + arm*twist

        return Z
        
        
    def _interp(self, x, y, xq):
        """
        linear interpolation of y vs x, to compute yq at xq locations

        """   
        
        if self.complex_step is True:
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
                
        
    def _RBF_TPS(self, x, y, xq):
        """
        RBF/TPS interpolation of y vs x, to compute yq at xq locations

        """   
        
        R = np.sqrt((np.tile(x,[len(x),1]).transpose() - np.tile(x,[len(x),1]))**2)
        a = np.argwhere(R==0)
        R[a[:,0],a[:,1]] = 100
        R = R**2.*np.log(R)
        R[a[:,0],a[:,1]] = 0
        P = np.c_[x,x*0+1]
        boo = np.linalg.solve(np.r_[np.c_[R,P],np.c_[P.transpose(),np.zeros([2,2])]],np.r_[y,0,0])
        
        R = np.sqrt((np.tile(xq,[len(x),1]).transpose() - np.tile(x,[len(xq),1]))**2)
        a = np.argwhere(R==0)
        R[a[:,0],a[:,1]] = 100
        R = R**2.*np.log(R)
        R[a[:,0],a[:,1]] = 0
        yq = R@boo[0:-2] + xq*boo[-2] + boo[-1]
        
        return yq
    



## function which creates the side-limits for the airfoil thickness DVs, as a fraction of the baseline thickness
    
def airfoil_thickness_bounds(xs,y_knot,airfoil_thickness_fraction):
    
    ## unpack coordinates
        
    X = xs[0::3]
    Y = xs[1::3]
    Z = xs[2::3]
        
    ## compute wing thickness
    
    thickness = np.zeros(len(y_knot))
    
    for i in range(0,len(y_knot)):
        fs = np.argwhere(np.abs(Y - y_knot[i]) < np.max(Y)/200.)
        thickness[i] = np.max(Z[fs]) - np.min(Z[fs])
    
    ## final bounds on thickness DVs
    
    thickness_min = -thickness*airfoil_thickness_fraction
    thickness_min = thickness_min[1:]
    
    thickness_max = thickness*airfoil_thickness_fraction
    thickness_max = thickness_max[1:]
    
    return thickness_min, thickness_max
