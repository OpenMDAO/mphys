from mpi4py import MPI
import openmdao.api as om

from mphys import Multipoint
from mphys.mphys_vlm import VlmBuilder
from mphys.scenario_aero import ScenarioAero


class Top(Multipoint):
    def setup(self):
        # VLM options
        mesh_file = 'wing_VLM.dat'

        mach = 0.85,
        aoa = 0.0
        q_inf = 3000.
        vel = 178.
        mu = 3.5E-5

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('mu', mu)

        aero_builder = VlmBuilder(mesh_file)
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('cruise',ScenarioAero(aero_builder=aero_builder))
        self.connect('mesh.x_aero0','cruise.x_aero')

        for dv in ['aoa', 'mach', 'q_inf', 'vel', 'mu']:
            self.connect(dv, f'cruise.{dv}')

prob = om.Problem()
prob.model = Top()
prob.setup()

om.n2(prob, show_browser=False, outfile='vlm_aero.html')

prob.run_model()
if MPI.COMM_WORLD.rank == 0:
    print('C_L, C_D =',prob['cruise.C_L'], prob['cruise.C_D'])