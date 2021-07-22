import numpy as np

## create a function which takes in elemental dv_struct, X, Y, and Z, 
##  and computes A/B/D, strain-to-stress matrix, and mass of each element

def dv_map(dv_struct_el, Xel, Yel, Zel):

    ## material properties
  
    E_modulus = 73.1e9
    poisson = 0.33
    rho = 2780.
    YS = 420e6
    
    ## stress-strain Q matrix

    Q = (E_modulus/(1-poisson**2))*np.array([[1,poisson,0],[poisson,1,0],[0,0,(1-poisson)/2]])
    
    ## allocate

    if np.iscomplexobj(dv_struct_el) != 0 or np.iscomplexobj(np.c_[Xel, Yel, Zel]) != 0:
        ABD = np.zeros([6,6,len(dv_struct_el)],dtype=complex)
        stress_strain = np.zeros([3,6,len(dv_struct_el)],dtype=complex)
    else:
        ABD = np.zeros([6,6,len(dv_struct_el)])
        stress_strain = np.zeros([3,6,len(dv_struct_el)])
        
    ## A and D matrices

    ABD[0:3,0:3,:] = np.tile(dv_struct_el[:,0],[3,3,1])*np.dstack([Q]*len(dv_struct_el))
    ABD[3:6,3:6,:] = np.tile(dv_struct_el[:,0]*dv_struct_el[:,0]*dv_struct_el[:,0]/12,[3,3,1])*np.dstack([Q]*len(dv_struct_el))
    
    ## stress-strain relationship

    stress_strain[:,0:3,:] = np.dstack([Q]*len(dv_struct_el))/YS
    
    ## mass per area

    mass_per_area = rho*dv_struct_el[:,0]
            
    return ABD, stress_strain, mass_per_area

