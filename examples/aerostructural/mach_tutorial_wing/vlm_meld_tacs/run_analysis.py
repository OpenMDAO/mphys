#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.mphys_vlm import VlmBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_meld import MeldBuilder

import tacs_setup
from tacs import functions

class Top(Multipoint):
    def setup(self):
        # VLM
        mesh_file = 'wing_VLM.dat'
        mach = 0.85
        aoa0 = 2.0
        aoa1 = 5.0
        q_inf = 3000.
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

        # TACS
        tacs_options = {'element_callback': tacs_setup.element_callback,
                        'mesh_file': 'wingbox_Y_Z_flip.bdf'}

        struct_builder = TacsBuilder(tacs_options, coupled=True)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem())

        dvs.add_output('dv_struct', np.array(ndv_struct*[0.002]))

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

            for dv in ['q_inf','vel','nu','mach', 'dv_struct']:
                self.connect(dv, f'{scenario}.{dv}')
            self.connect('aoa', f'{scenario}.aoa', src_indices=[iscen])

    def configure(self):
        for scenario_name in ['cruise','maneuver']:
            scenario = getattr(self, scenario_name)
            fea_solver = scenario.coupling.struct.fea_solver

            # ==============================================================================
            # Setup structural problem
            # ==============================================================================
            # Structural problem
            sp = fea_solver.createStaticProblem(name=scenario_name)
            # Add TACS Functions
            sp.addFunction('mass', functions.StructuralMass)
            sp.addFunction('ks_vmfailure', functions.KSFailure, ksWeight=50.0, safetyFactor=1.0)

            scenario.coupling.struct.mphys_set_sp(sp)
            scenario.struct_post.mphys_set_sp(sp)

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
om.n2(prob, show_browser=False, outfile='mphys_as_vlm.html')

prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print('C_L =',prob['cruise.C_L'])
    print('KS =',prob['maneuver.struct_post.ks_vmfailure'])
