import numpy as np
from tacs import constitutive, elements, functions


# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    rho = 2780.0  # density, kg/m^3
    E = 73.1e9  # elastic modulus, Pa
    nu = 0.33  # poisson's ratio
    ys = 324.0e6  # yield stress, Pa
    thickness = 0.003
    min_thickness = 0.002
    max_thickness = 0.05

    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=thickness, tNum=dvNum, tlb=min_thickness, tub=max_thickness)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None

    elemList = []
    for elemDescript in elemDescripts:
        if elemDescript in ["CQUAD4", "CQUADR"]:
            elem = elements.Quad4Shell(transform, con)
        elif elemDescript in ["CTRIA3", "CTRIAR"]:
            elem = elements.Tri3Shell(transform, con)
        else:
            raise ValueError("Uh oh, '%s' not recognized" % (elemDescript))
        elemList.append(elem)

    return elemList

def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Add TACS Functions
    # Only include mass from elements that belong to pytacs components (i.e. skip concentrated masses)
    problem.addFunction('mass', functions.StructuralMass)
    problem.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

    # Add 2.5G gravity load
    g = np.array([0.0, 0.0, -9.81])  # m/s^2
    problem.addInertialLoad(2.5*g)

def constraint_setup(scenario_name, fea_assembler, constraint_list):
    """
    Helper function to setup tacs constraint classes
    """
    if scenario_name == "maneuver":
        # Setup adjacency constraints for skin and spar panel thicknesses
        constr = fea_assembler.createAdjacencyConstraint("adjacency")
        compIDs = fea_assembler.selectCompIDs(include="UPPER_SKIN")
        constr.addConstraint("UPPER_SKIN", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="LOWER_SKIN")
        constr.addConstraint("LOWER_SKIN", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="LE_SPAR")
        constr.addConstraint("LE_SPAR", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="TE_SPAR")
        constr.addConstraint("TE_SPAR", compIDs=compIDs)
        constraint_list.append(constr)

