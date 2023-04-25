import numpy as np
from tacs import elements, constitutive, functions

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

    # Define reference axis for local shell stresses
    if 'SKIN' in compDescript: # USKIN + LSKIN
        sweep = 35.0 / 180.0 * np.pi
        refAxis = np.array([np.sin(sweep), np.cos(sweep), 0])
    else: # RIBS + SPARS + ENGINE_MOUNT
        refAxis = np.array([0.0, 0.0, 1.0])

    # For each element type in this component,
    # pass back the appropriate tacs element object
    elemList = []
    transform = elements.ShellRefAxisTransform(refAxis)
    for elemDescript in elemDescripts:
        if elemDescript in ['CQUAD4', 'CQUADR']:
            elem = elements.Quad4Shell(transform, con)
        elif elemDescript in ['CTRIA3', 'CTRIAR']:
            elem = elements.Tri3Shell(transform, con)
        else:
            print("Uh oh, '%s' not recognized" % (elemDescript))
        elemList.append(elem)

    # Add scale for thickness dv
    scale = [100.0]
    return elemList, scale

def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """

    # Add TACS Functions
    problem.addFunction('mass', functions.StructuralMass)
    problem.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0,
                        ksWeight=100.0)

    # Add forces to static problem
    F = fea_assembler.createVec()
    ndof = fea_assembler.getVarsPerNode()
    F[2::ndof] = 100.0
    problem.addLoadToRHS(F)

def constraint_setup(scenario_name, fea_assembler, constraint_list):
    """
    Helper function to setup tacs constraint classes
    """
    if scenario_name == "analysis":
        # Setup adjacency constraints for skin and spar panel thicknesses
        constr = fea_assembler.createAdjacencyConstraint("adjacency")
        compIDs = fea_assembler.selectCompIDs(include="U_SKIN")
        constr.addConstraint("U_SKIN", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="L_SKIN")
        constr.addConstraint("L_SKIN", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="LE_SPAR")
        constr.addConstraint("LE_SPAR", compIDs=compIDs)
        compIDs = fea_assembler.selectCompIDs(include="TE_SPAR")
        constr.addConstraint("TE_SPAR", compIDs=compIDs)
        constraint_list.append(constr)

