import numpy as np
import openmdao.api as om

from mphys.multipoint import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from vlm_solver.mphys_vlm import VlmBuilder
from tacs.mphys import TacsBuilder
from funtofem.mphys import MeldBuilder

from struct_dv_components import StructDvMapper, SmoothnessEvaluatorGrid, struct_comps
import tacs_setup

check_derivs = False
class Top(Multipoint):
    def setup(self):
        # VLM
        mesh_file = 'wing_VLM.dat'
        mach = 0.85
        aoa0 = 2.0
        aoa1 = 5.0
        q_inf = 12000.
        vel = 178.
        nu = 3.5E-5

        aero_builder = VlmBuilder(mesh_file)
        aero_builder.initialize(self.comm)

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=[aoa0,aoa1], units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('nu', nu)

        self.add_subsystem('mesh_aero',aero_builder.get_mesh_coordinate_subsystem())

        # TACS setup
        struct_builder = TacsBuilder(mesh_file='wingbox_Y_Z_flip.bdf', element_callback= tacs_setup.element_callback,
                                     problem_setup= tacs_setup.problem_setup)
        struct_builder.initialize(self.comm)

        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem())

        initial_thickness = 0.003
        dvs.add_output('ribs',        val=initial_thickness, shape = struct_comps['ribs'])
        dvs.add_output('le_spar',     val=initial_thickness, shape = struct_comps['le_spar'])
        dvs.add_output('te_spar',     val=initial_thickness, shape = struct_comps['te_spar'])
        dvs.add_output('up_skin',     val=initial_thickness, shape = struct_comps['up_skin'])
        dvs.add_output('lo_skin',     val=initial_thickness, shape = struct_comps['lo_skin'])
        dvs.add_output('up_stringer', val=initial_thickness, shape = struct_comps['up_stringer'])
        dvs.add_output('lo_stringer', val=initial_thickness, shape = struct_comps['lo_stringer'])

        self.add_subsystem('struct_mapper',StructDvMapper(), promotes=['*'])

        # MELD setup
        isym = 1
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)
        ldxfer_builder.initialize(self.comm)

        for iscen, scenario in enumerate(['cruise','maneuver']):
            nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1E-14, atol=1E-14)
            linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-14, atol=1e-14)
            self.mphys_add_scenario(scenario,ScenarioAeroStructural(aero_builder=aero_builder,
                                                                    struct_builder=struct_builder,
                                                                    ldxfer_builder=ldxfer_builder),
                                             nonlinear_solver, linear_solver)

            for discipline in ['aero','struct']:
                self.mphys_connect_scenario_coordinate_source('mesh_%s' % discipline, scenario, discipline)

            for dv in ['q_inf','vel','nu','mach','dv_struct']:
                self.connect(dv, f'{scenario}.{dv}')
            self.connect('aoa', f'{scenario}.aoa', src_indices=[iscen])

        self.add_subsystem('le_spar_smoothness',SmoothnessEvaluatorGrid(columns=struct_comps['le_spar'],rows=1))
        self.add_subsystem('te_spar_smoothness',SmoothnessEvaluatorGrid(columns=struct_comps['te_spar'],rows=1))
        self.add_subsystem('up_skin_smoothness',SmoothnessEvaluatorGrid(columns=9,rows=struct_comps['up_skin']//9))
        self.add_subsystem('lo_skin_smoothness',SmoothnessEvaluatorGrid(columns=9,rows=int(struct_comps['lo_skin']/9)))

        self.connect('le_spar','le_spar_smoothness.thickness')
        self.connect('te_spar','te_spar_smoothness.thickness')
        self.connect('up_skin','up_skin_smoothness.thickness')
        self.connect('lo_skin','lo_skin_smoothness.thickness')

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model

# optimization set up
prob.model.add_design_var('aoa',lower=-5*np.pi/180, upper=10*np.pi/180.0, ref=1.0, units='rad')
prob.model.add_design_var('ribs',        lower=0.001, upper=0.020, ref=0.005)
prob.model.add_design_var('le_spar',     lower=0.001, upper=0.020, ref=0.005)
prob.model.add_design_var('te_spar',     lower=0.001, upper=0.020, ref=0.005)
prob.model.add_design_var('up_skin',     lower=0.001, upper=0.020, ref=0.005)
prob.model.add_design_var('lo_skin',     lower=0.001, upper=0.020, ref=0.005)
prob.model.add_design_var('up_stringer', lower=0.001, upper=0.020, ref=0.005)
prob.model.add_design_var('lo_stringer', lower=0.001, upper=0.020, ref=0.005)

prob.model.add_objective('cruise.mass',ref=1000.0)
prob.model.add_constraint('cruise.C_L',ref=1.0,equals=0.5)
prob.model.add_constraint('maneuver.C_L',ref=1.0,equals=0.9)
prob.model.add_constraint('maneuver.ks_vmfailure',ref=1.0, upper = 2.0/3.0)

prob.model.add_constraint('le_spar_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)
prob.model.add_constraint('te_spar_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)
prob.model.add_constraint('up_skin_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)
prob.model.add_constraint('lo_skin_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)

#prob.driver = om.ScipyOptimizeDriver(debug_print=['ln_cons','nl_cons','objs','totals'])
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-8
prob.driver.options['disp'] = True

prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True

recorder = om.SqliteRecorder("cases.sql")
prob.driver.add_recorder(recorder)

prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='mphys_as_vlm.html')

if check_derivs:
    prob.run_model()
    prob.check_totals(of=['cruise.mass','cruise.C_L','maneuver.ks_vmfailure'],
                      wrt=['aoa','ribs'])
else:
    prob.run_driver()
    cr = om.CaseReader('cases.sql')
    driver_cases = cr.list_cases('driver')

    matrix = np.zeros((len(driver_cases),4))
    for i, case_id in enumerate(driver_cases):
        matrix[i,0] = i
        case = cr.get_case(case_id)
        matrix[i,1] = case.get_objectives()['cruise.mass'][0]
        matrix[i,2] = case.get_constraints()['cruise.C_L'][0]
        matrix[i,3] = case.get_constraints()['maneuver.ks_vmfailure'][0]
    np.savetxt('history.dat',matrix)
