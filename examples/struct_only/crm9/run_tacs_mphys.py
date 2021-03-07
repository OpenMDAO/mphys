"""
Mass minimization of uCRM wingbox subject to a constant vertical force

"""
from __future__ import division, print_function
import numpy as np

import openmdao.api as om

from mphys.mphys_tacs import TacsBuilder, TacsMesh
from mphys import Multipoint
from mphys.scenario_structural import ScenarioStructural
import tacs_setup

class Top(Multipoint):
    def setup(self):
        tacs_options = {'add_elements' : tacs_setup.add_elements,
                        'mesh_file'    : 'CRM_box_2nd.bdf',
                        'get_funcs'    : tacs_setup.get_funcs,
                        'load_function': tacs_setup.forcer_function,
                        'f5_writer'    : tacs_setup.f5_writer}

        struct_builder = TacsBuilder(tacs_options)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct',np.array(ndv_struct*[0.0031]))

        self.add_subsystem('mesh',struct_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=struct_builder))
        self.mphys_connect_scenario_coordinate_source('mesh','analysis','struct')

        self.connect('dv_struct', 'analysis.dv_struct')

################################################################################
# OpenMDAO setup
################################################################################

prob = om.Problem()
prob.model = Top()
model = prob.model

model.add_design_var('dv_struct',lower=0.001,upper=0.075,scaler=1000.0)
model.add_objective('analysis.mass', index=0, scaler=1.0/1000.0)
model.add_constraint('analysis.func_struct', indices=[0], lower = 0.0, upper = 2.0/3.0,scaler=1.0)

prob.driver = om.ScipyOptimizeDriver(debug_print=['objs','nl_cons'],maxiter=1)
prob.driver.options['optimizer'] = 'SLSQP'

prob.setup()
om.n2(prob, show_browser=False, outfile='tacs_struct.html')
prob.run_model()

prob.run_driver()
for i in range(240):
    print('final dvs',i,prob['dv_struct'][i])
