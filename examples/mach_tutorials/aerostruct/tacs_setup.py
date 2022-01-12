import numpy as np
from tacs import elements, constitutive

# Material properties
rho = 2500.0        # density kg/m^3
E = 70.0e9          # Young's modulus (Pa)
nu = 0.30           # Poisson's ratio
kcorr = 5.0/6.0     # shear correction factor
ys = 350e6        # yield stress

# Shell thickness
t = 0.01            # m
tMin = 0.002        # m
tMax = 0.05         # m

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None
    elem = elements.Quad4Shell(transform, con)
    return elem
