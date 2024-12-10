"""
Mass minimization of a thin-walled ellipsoidal tank with a volume constraint.
Design variables are the semi-axes of the ellipsoid.
"""
from __future__ import division, print_function
import numpy as np

import openmdao.api as om

from tacs.mphys import TacsBuilder
from pygeo.mphys import OM_DVGEOCOMP
from mphys import Multipoint
from mphys.scenarios.structural import ScenarioStructural
from tacs import functions

bdf_file = 'tank.dat'
vsp_file = "tank.vsp3"

target_vol = 4.0/3.0 * np.pi

def problem_setup(scenario_name, fea_assembler, problem):
    problem.addFunction("mass", functions.StructuralMass)

def constraint_setup(scenario_name, fea_assembler, constraints):
    constr = fea_assembler.createVolumeConstraint("constraints")
    constr.addConstraint("volume")
    constraints.append(constr)

class Top(Multipoint):
    def setup(self):
        tacs_options = {'mesh_file': bdf_file,
                        'problem_setup': problem_setup,
                        'constraint_setup': constraint_setup}

        struct_builder = TacsBuilder(mesh_file=bdf_file, problem_setup=problem_setup,
                                     constraint_setup=constraint_setup, coupled=False)
        struct_builder.initialize(self.comm)

        # Add mesh component
        self.add_subsystem('mesh', struct_builder.get_mesh_coordinate_subsystem())

        # add the geometry component, we dont need a builder because we do it here.
        self.add_subsystem("geometry", OM_DVGEOCOMP(file=vsp_file, type="vsp"))
        self.geometry.nom_add_discipline_coords("struct")

        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=struct_builder))

        self.connect("mesh.x_struct0", "geometry.x_struct_in")
        self.connect("geometry.x_struct0", "analysis.x_struct0")

    def configure(self):
        # create geometric DV setup
        self.geometry.nom_addVSPVariable('Tank', 'Design', 'A_Radius', scaledStep=False)
        self.geometry.nom_addVSPVariable('Tank', 'Design', 'B_Radius', scaledStep=False)
        self.geometry.nom_addVSPVariable('Tank', 'Design', 'C_Radius', scaledStep=False)

################################################################################
# OpenMDAO setup
################################################################################

prob = om.Problem()
prob.model = Top()
model = prob.model

# Declare design variables, objective, and constraint
model.add_design_var('geometry.Tank:Design:A_Radius', lower=1e-3, upper=10.0, scaler=20.0)
model.add_design_var('geometry.Tank:Design:B_Radius', lower=1e-3, upper=10.0, scaler=20.0)
model.add_design_var('geometry.Tank:Design:C_Radius', lower=1e-3, upper=10.0, scaler=20.0)
model.add_objective('analysis.mass', scaler=1e-2)
model.add_constraint('analysis.constraints.volume', equals=target_vol, scaler=1.0)

# Configure optimizer
prob.driver = om.ScipyOptimizeDriver(debug_print=['objs', 'nl_cons'], maxiter=100)
prob.driver.options['optimizer'] = 'SLSQP'

prob.setup()
om.n2(prob, show_browser=False, outfile='tank_prob.html')
prob.run_driver()
