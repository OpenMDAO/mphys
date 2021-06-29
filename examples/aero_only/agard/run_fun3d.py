
from mpi4py import MPI
import openmdao.api as om

from mphys import Multipoint
from mphys.mphys_fun3d import Fun3dSfeBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic


class Top(Multipoint):
    def setup(self):
        # FUN3D options
        boundary_tags = [3]

        mach = 0.5,
        aoa = 1.0
        yaw = 0.0
        q_inf = 3000.
        reynolds = 0.0

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('yaw', val=yaw, units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('reynolds', reynolds)

        aero_builder = Fun3dSfeBuilder(boundary_tags)
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('cruise',ScenarioAerodynamic(aero_builder=aero_builder))
        self.connect('mesh.x_aero0','cruise.x_aero')

        for dv in ['aoa', 'yaw', 'mach', 'q_inf','reynolds']:
            self.connect(dv, f'cruise.{dv}')

prob = om.Problem()
prob.model = Top()
prob.setup(mode='rev')

om.n2(prob, show_browser=False, outfile='mphys_fun3d.html')

prob.run_model()
if MPI.COMM_WORLD.rank == 0:
    print('C_L, C_D =',prob['cruise.C_L'], prob['cruise.C_D'], prob['cruise.C_X'],prob['cruise.C_Z'])

prob.check_totals(of='cruise.C_L',wrt='aoa')
