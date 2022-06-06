"""
This example demonstrates TACS structural optimization capabilities.
The beam model that we will be using for this problem is a rectangular beam,
cantilevered, with a shear load applied at the tip. The beam is discretized using
1001 shell elements along it's span and depth.

The optimization problem is as follows:
Minimize the mass of the beam with respect to the dpeth of the cross-section along the span,
subject to a max stress constraint dictated by the materials yield stress.

In order to change the shape of the FEM we use a free-form deformation (FFD) volume
parmeterization scheme provided by the pyGeo library.

An aproximate analytcal solution can be derived from beam theory,
by realizing that the stress at any spanwise cross-section in the beam
can be found independently using:
    sigma(x,y) = y*M(x)/I
An analytical solution for this problem can be shown to be:
    t(x) = sqrt(6*V*(L-x)/(t*sigma_y))

The optimization is setup using TACS' MPHYS module, which acts as a wrapper
for OpenMDAO.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import openmdao.api as om
from pygeo.mphys import OM_DVGEOCOMP

from mphys import Multipoint
from mphys.scenario_structural import ScenarioStructural
from tacs.mphys import TacsBuilder
from tacs import elements, constitutive, functions

bdf_file = os.path.join(os.path.dirname(__file__), 'Slender_Beam.bdf')
ffd_file = os.path.join(os.path.dirname(__file__), 'ffd_8_linear.fmt')

# Beam thickness
t = 0.01            # m
# Length of beam
L = 1.0

# Material properties
rho = 2780.0 # kg /m^3
E = 70.0e9
nu = 0.0
ys = 420.0e6

# Shear force applied at tip
V = 2.5E4

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    con = constitutive.IsoShellConstitutive(prop, t=t, tNum=-1)
    # TACS shells are sometimes a little overly-rigid in shear
    # We can reduce this effect by decreasing the drilling regularization
    con.setDrillingRegularization(0.1)
    refAxis = np.array([1.0, 0.0, 0.0])
    transform = elements.ShellRefAxisTransform(refAxis)
    elem = elements.Quad4Shell(transform, con)
    return elem

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
    problem.addLoadToNodes(1112, [0.0, V, 0.0, 0.0, 0.0, 0.0], nastranOrdering=True)


class Top(Multipoint):
    def setup(self):
        tacs_options = {'element_callback': element_callback,
                        'problem_setup': problem_setup,
                        'mesh_file': bdf_file}

        # Initialize MPHYS builder for TACS
        struct_builder = TacsBuilder(tacs_options, coupled=False)
        struct_builder.initialize(self.comm)

        # Add mesh component
        self.add_subsystem('mesh', struct_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(ffd_file=ffd_file))
        self.geometry.nom_add_discipline_coords("struct")

        self.mphys_add_scenario('tip_shear', ScenarioStructural(struct_builder=struct_builder))

        self.connect("mesh.x_struct0", "geometry.x_struct_in")
        self.connect("geometry.x_struct0", "tip_shear.x_struct0")

    def configure(self):
        # Create reference axis
        nRefAxPts = self.geometry.nom_addRefAxis(name="centerline", alignIndex='i', yFraction=0.5)

        # Set up global design variables
        def depth(val, geo):
            for i in range(nRefAxPts):
                geo.scale_y["centerline"].coef[i] = val[i]

        self.geometry.nom_addGeoDVGlobal(dvName="depth", value=np.ones(nRefAxPts), func=depth)


################################################################################
# OpenMDAO setup
################################################################################
# Instantiate OpenMDAO problem
prob = om.Problem()
prob.model = Top()
model = prob.model

# Declare design variables, objective, and constraint
model.add_design_var('geometry.depth', lower=1e-3, upper=10.0, scaler=20.0)
model.add_objective('tip_shear.mass', scaler=1.0)
model.add_constraint('tip_shear.ks_vmfailure', lower=0.0, upper=1.0, scaler=1.0)

# Configure optimizer
prob.driver = om.ScipyOptimizeDriver(debug_print=['objs', 'nl_cons'], maxiter=1000)
prob.driver.options['optimizer'] = 'SLSQP'

# Setup OpenMDAO problem
prob.setup()

# Output N2 representation of OpenMDAO model
om.n2(prob, show_browser=False, outfile='beam_opt_n2.html')

# Run optimization
prob.run_driver()
