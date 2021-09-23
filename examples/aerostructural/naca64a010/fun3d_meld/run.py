#rst Imports
from __future__ import print_function, division
import numpy as np

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.mphys_fun3d_lfd import Fun3dLfdBuilder
from mphys.mphys_meld_lfd import MeldLfdBuilder
from isogai_struct import IsogaiStructBuilder
from speed_index import SpeedIndexComp


class Top(Multipoint):
    def setup(self):
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        aoa = 0.0
        mach = 0.75

        semichord = 0.5
        rho_ref = 1.225
        u_ref = 400
        q_inf = 0.5 * rho_ref * u_ref ** 2.0

        # LFD options
        nmodes = 2
        pk_density = np.linspace(rho_ref-1e-4,rho_ref+1e-4)
        pk_velocity = np.linspace(0.4 * u_ref, 10.0 * u_ref)
        reduced_frequencies = np.array([0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

        # FUN3D options
        boundary_tag_list = [1]
        aero_builder = Fun3dLfdBuilder(boundary_tag_list, nmodes, u_ref, semichord, reduced_frequencies, pk_density, pk_velocity, input_file='input.cfg')
        aero_builder.initialize(self.comm)

        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('yaw', val=0.0, units='deg')
        dvs.add_output('mach', val=mach)
        dvs.add_output('reynolds', val=0.0)
        dvs.add_output('q_inf', val=q_inf)
        aero_dvs = ['aoa','mach','reynolds','q_inf','yaw']

        struct_builder = IsogaiStructBuilder()
        struct_builder.initialize(self.comm)

        dvs.add_output('ref_length', val = semichord)
        dvs.add_output('pitch_frequency', val=100.0)
        dvs.add_output('plunge_frequency', val=100.0)
        struct_dvs = ['pitch_frequency','plunge_frequency', 'ref_length']

        # Transfer scheme options
        isym = -1
        ldxfer_builder = MeldLfdBuilder(aero_builder, struct_builder, nmodes, isym=isym, check_partials=True)
        ldxfer_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero',aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem())

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8, aitken_min_factor = 0.0000001, aitken_max_factor = 0.1)
        self.mphys_add_scenario('flutter',ScenarioAeroStructural(aero_builder=aero_builder,
                                                                struct_builder=struct_builder,
                                                                ldxfer_builder=ldxfer_builder),
                                         nonlinear_solver, linear_solver)
        self.add_subsystem('speed_index',SpeedIndexComp(),promotes=['*'])

        for discipline in ['aero','struct']:
            self.mphys_connect_scenario_coordinate_source('mesh_%s' % discipline, 'flutter', discipline)

        for dv in aero_dvs:
            self.connect(dv, f'flutter.{dv}')
        for dv in struct_dvs:
            self.connect(dv, f'flutter.{dv}')
        self.connect('flutter.flutter_q','flutter_q')


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='crm_aerostruct.html')


prob.run_model()
output = prob.check_totals(of=['vf'], wrt=['mach'])
