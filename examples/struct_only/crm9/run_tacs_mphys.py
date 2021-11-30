"""
Mass minimization of uCRM wingbox subject to a constant vertical force
"""
from __future__ import division, print_function

import openmdao.api as om

from mphys.mphys_tacs import TacsBuilder
from mphys import Multipoint
from mphys.scenario_structural import ScenarioStructural
import tacs_setup
from tacs import functions


class Top(Multipoint):
    def setup(self):
        tacs_options = {'element_callback': tacs_setup.element_callback,
                        'mesh_file': 'CRM_box_2nd.bdf'}

        struct_builder = TacsBuilder(tacs_options, check_partials=True)
        struct_builder.initialize(self.comm)
        dv_array = struct_builder.get_initial_dvs()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', dv_array)

        self.add_subsystem('mesh', struct_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=struct_builder))
        self.mphys_connect_scenario_coordinate_source('mesh', 'analysis', 'struct')

        self.connect('dv_struct', 'analysis.dv_struct')

    def configure(self):
        # create the tacs problems for analysis point.
        # this is custom to the tacs based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        fea_solver = self.analysis.coupling.fea_solver

        # ==============================================================================
        # Setup static problem
        # ==============================================================================
        # Static problem
        sp = fea_solver.createStaticProblem(name='analysis', options={'printTiming':True})
        # Add TACS Functions
        sp.addFunction('mass', functions.StructuralMass)
        sp.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0,
                       ksWeight=100.0)

        # Add forces to static problem
        F = fea_solver.createVec()
        ndof = fea_solver.getVarsPerNode()
        F[2::ndof] = 100.0
        sp.addLoadToRHS(F)

        # here we set the tacs problems for the analysis case we have.
        # this call automatically adds the functions and loads for the respective scenario
        self.analysis.coupling.mphys_set_sp(sp)
        self.analysis.struct_post.mphys_set_sp(sp)


################################################################################
# OpenMDAO setup
################################################################################

prob = om.Problem()
prob.model = Top()
model = prob.model

model.add_design_var('dv_struct', lower=0.002, upper=0.2, scaler=1000.0)
model.add_objective('analysis.struct_post.mass', index=0, scaler=1.0 / 1000.0)
model.add_constraint('analysis.struct_post.ks_vmfailure', lower=0.0, upper=1.0, scaler=1.0)

prob.driver = om.ScipyOptimizeDriver(debug_print=['objs', 'nl_cons'], maxiter=1)
prob.driver.options['optimizer'] = 'SLSQP'

prob.setup()
om.n2(prob, show_browser=False, outfile='tacs_struct.html')
prob.run_model()

prob.run_driver()
for i in range(242):
    print('final dvs', i, prob['dv_struct'][i])
