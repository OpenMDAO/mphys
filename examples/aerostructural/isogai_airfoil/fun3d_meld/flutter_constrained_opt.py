import numpy as np
import openmdao.api as om

from mphys import Multipoint
from mphys.mphys_fun3d_lfd import Fun3dLfdBuilder
from mphys.mphys_meld_lfd import MeldLfdBuilder
from isogai_struct import IsogaiStructBuilder
from mphys.scenario_aerostructural import ScenarioAeroStructural

from pyffd.om_twod_ffd import FFD2DBuilder

from geometry_sum import GeometrySum

mach = .875
u_ref = 400.0
rho_ref = 1.225

aoa = 0.0

semichord = 0.5
q_inf = 100000.

alpha_m = .0
alpha_k = 0.00

pk_density = np.linspace(rho_ref-1e-4,rho_ref+1e-4)
pk_velocity = np.linspace(0.4 * u_ref, 4.0 * u_ref)

KS_parameter = 200.0
flutter_q_bound = 694804.6875
flutter_vel_bound = np.sqrt(flutter_q_bound*2/np.mean(pk_density))
flutter_g_bound = -.25
flutter_bound_coeffs = np.c_[np.array([-2,3*flutter_vel_bound,0,0])*(flutter_g_bound/flutter_vel_bound**3.),np.array([0,1/100.0,0,flutter_g_bound])]
flutter_bound_breaks = np.array([0.,flutter_vel_bound,np.max(pk_velocity)])

class Top(Multipoint):
    def setup(self):
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # LFD options
        nmodes = 2
        reduced_frequencies = np.array([0.0, 0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])

        # FUN3D options
        boundary_tag_list = [1]
        aero_builder = Fun3dLfdBuilder(boundary_tag_list, nmodes, u_ref, semichord,
                                       reduced_frequencies, pk_density, pk_velocity, input_file='input.cfg',
                                       flutter_bound_coeffs=flutter_bound_coeffs,
                                       flutter_bound_breaks=flutter_bound_breaks,
                                       KS_parameter=KS_parameter)
        aero_builder.initialize(self.comm)

        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('yaw', val=0.0, units='deg')
        dvs.add_output('mach', val=mach)
        dvs.add_output('reynolds', val=0.0)
        dvs.add_output('q_inf', val=q_inf)

        struct_builder = IsogaiStructBuilder()
        struct_builder.initialize(self.comm)

        dvs.add_output('ref_length', val = semichord)
        dvs.add_output('pitch_frequency', val=100.0)
        dvs.add_output('plunge_frequency', val=100.0)
        dvs.add_output('alpha_m', val=alpha_m)
        dvs.add_output('alpha_k', val=alpha_k)

        # Geometry
        builders = {'aero': aero_builder}
        x_bounds = [-.1, 1.1]
        z_bounds = [-0.06, 0.06]
        geom_builder = FFD2DBuilder(builders, x_bounds, z_bounds, idim=8, jdim=3)

        # Transfer scheme options
        isym = -1
        ldxfer_builder = MeldLfdBuilder(aero_builder, struct_builder, nmodes, isym=isym, check_partials=True)
        ldxfer_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero',aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem(),promotes=['*'])
        self.add_subsystem('geom', geom_builder.get_mesh_coordinate_subsystem(), promotes=['*'])

        dvs.add_output('ctrl_pt_delta', np.zeros_like(geom_builder.control_points))

        self.connect('mesh_aero.x_aero0', 'x_aero0_in')

        nonlinear_solver = om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8)
        linear_solver = om.LinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8)
        self.mphys_add_scenario('flutter',ScenarioAeroStructural(aero_builder=aero_builder,
                                                                struct_builder=struct_builder,
                                                                ldxfer_builder=ldxfer_builder),
                                         nonlinear_solver, linear_solver, promotes=True)
        self.add_subsystem('geometry_sum', GeometrySum(), promotes=['*'])

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model

prob.driver = om.ScipyOptimizeDriver(debug_print=['desvars','ln_cons','nl_cons','objs','totals'])
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-5
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 200

prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True

recorder = om.SqliteRecorder("cases_isogai_flutter.sql")
prob.driver.add_recorder(recorder)

model.add_design_var('ctrl_pt_delta', lower=-0.075, upper=0.075, ref =.001)

model.add_objective('total_movement', ref=1.0)
model.add_constraint('flutter_KS', upper=0.0, ref = 1.0)

prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='isogai_opt.html')


prob.run_driver()
