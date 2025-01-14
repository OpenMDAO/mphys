import numpy as np
from tacs import constitutive, elements, functions

# Material properties
rho = 2500.0  # density kg/m^3
E = 70.0e9  # Young's modulus (Pa)
nu = 0.30  # Poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 350e6  # yield stress

# Shell thickness
t = 0.01  # m
tMin = 0.002  # m
tMax = 0.05  # m

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


def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """

    # Add TACS Functions
    if scenario_name == "cruise":
        problem.addFunction("mass", functions.StructuralMass)
    problem.addFunction(
        "ks_vmfailure", functions.KSFailure, safetyFactor=1.0, ksWeight=100.0
    )

    # Add gravity load
    g = np.array([0.0, 0.0, -9.81])  # m/s^2
    if scenario_name == "maneuver":
        problem.addInertialLoad(2.5 * g)
    else:  # cruise
        problem.addInertialLoad(g)
