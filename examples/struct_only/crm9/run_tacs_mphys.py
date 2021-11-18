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
        dv_src_indices = struct_builder.get_dv_src_indices()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', dv_array)

        self.add_subsystem('mesh', struct_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=struct_builder))
        self.mphys_connect_scenario_coordinate_source('mesh', 'analysis', 'struct')

        self.connect('dv_struct', 'analysis.dv_struct', src_indices=dv_src_indices)

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        fea_solver = self.analysis.coupling.fea_solver

        # ==============================================================================
        # Setup structural problem
        # ==============================================================================
        # Structural problem
        sp = fea_solver.createStaticProblem(name='cruise')
        # Add TACS Functions
        sp.addFunction('mass', functions.StructuralMass)
        sp.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0,
                       ksWeight=100.0)
        '''
        # Various methods for adding loads to structural problem:
        # Let's model an engine load of 75kN weight, and 64kN thrust attached to spar
        compIDs = fea_solver.selectCompIDs(["WING_SPARS/LE_SPAR/SEG.16", "WING_SPARS/LE_SPAR/SEG.17"])
        We = 75.0e3  # N
        Te = 64.0e3  # N
        sp.addLoadToComponents(compIDs, [-Te, 0.0, -We, 0.0, 0.0, 0.0], averageLoad=True)
        # Next we'll approximate aerodynamic loads on upper/lower skin with a uniform traction
        L = 3e3  # N/m^2
        D = 150  # N/m^2
        tracVec = np.array([D, 0.0, L])
        compIDs = fea_solver.selectCompIDs(include='SKIN')
        sp.addTractionToComponents(compIDs, tracVec)
        # Finally, we can approximate fuel load by adding a pressure load to the lower skin
        P = 2e3  # N/m^2
        compIDs = fea_solver.selectCompIDs(include='L_SKIN')
        sp.addPressureToComponents(compIDs, P)
        '''
        F = sp.F.getArray()
        ndof = fea_solver.getVarsPerNode()
        F[2::ndof] = 100.0

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
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

#prob.driver = om.ScipyOptimizeDriver(debug_print=['objs', 'nl_cons'], maxiter=100)
#prob.driver.options['optimizer'] = 'SLSQP'

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.opt_settings['Major iterations limit'] = 100

prob.setup()
om.n2(prob, show_browser=False, outfile='tacs_struct.html')
prob.run_model()

prob.run_driver()
for i in range(240):
    print('final dvs', i, prob['dv_struct'][i])
