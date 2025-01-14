#rst Imports
from __future__ import division, print_function

import numpy as np
import openmdao.api as om
import tacs_setup
from funtofem.mphys import MeldBuilder
from mpi4py import MPI
from tacs.mphys import TacsBuilder
from vlm_solver.mphys_vlm import VlmBuilder

from mphys import MultipointParallel
from mphys.scenarios.aerostructural import ScenarioAeroStructural


class AerostructParallel(MultipointParallel):
    def setup(self):
        # VLM
        mesh_file = 'wing_VLM.dat'
        aero_builder = VlmBuilder(mesh_file)

        # TACS setup
        struct_builder = TacsBuilder(mesh_file='wingbox_Y_Z_flip.bdf', element_callback=tacs_setup.element_callback,
                                     problem_setup=tacs_setup.problem_setup,
                                     coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC])

        # MELD
        isym = 1
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)

        for scenario in ['cruise','maneuver']:
            nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1E-14, atol=1E-14)
            linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-14, atol=1e-14)
            self.mphys_add_scenario(scenario,ScenarioAeroStructural(aero_builder=aero_builder,
                                                                    struct_builder=struct_builder,
                                                                    ldxfer_builder=ldxfer_builder,
                                                                    in_MultipointParallel=True),
                                             nonlinear_solver, linear_solver)

class Top(om.Group):
    def setup(self):
        # VLM
        mach = 0.85,
        aoa0 = 2.0
        aoa1 = 5.0
        q_inf = 3000.
        vel = 178.
        nu = 3.5E-5

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=[aoa0,aoa1], units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('nu', nu)

        # TACS
        ndv_struct = 810
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.002]))

        self.add_subsystem('multipoint',AerostructParallel())

        for iscen, scenario in enumerate(['cruise','maneuver']):
            for dv in ['q_inf','vel','nu','mach', 'dv_struct']:
                self.connect(dv, f'multipoint.{scenario}.{dv}')
            self.connect('aoa', f'multipoint.{scenario}.aoa', src_indices=[iscen])

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
#om.n2(prob, show_browser=False, outfile='mphys_as_parallel.html')

prob.run_model()

ks = prob.get_val('multipoint.maneuver.ks_vmfailure',get_remote=True)
cl = prob.get_val('multipoint.cruise.C_L', get_remote=True)
if MPI.COMM_WORLD.rank == 0:
    print('C_L =', cl)
    print('KS =', ks)
