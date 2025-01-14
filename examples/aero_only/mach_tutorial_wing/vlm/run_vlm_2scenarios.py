import openmdao.api as om
from mpi4py import MPI
from vlm_solver.mphys_vlm import VlmBuilder

from mphys import Multipoint
from mphys.scenarios.aerodynamic import ScenarioAerodynamic


class Top(Multipoint):
    def setup(self):
        # VLM options
        mesh_file = 'wing_VLM.dat'

        mach0 = 0.85,
        mach1 = 0.80,
        aoa0 = 0.0
        aoa1 = 1.0
        q_inf = 3000.
        vel = 178.
        nu = 3.5E-5

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa0', val=aoa0, units='deg')
        dvs.add_output('aoa1', val=aoa1, units='deg')
        dvs.add_output('mach0', mach0)
        dvs.add_output('mach1', mach1)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('nu', nu)

        aero_builder = VlmBuilder(mesh_file)
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('cruise',ScenarioAerodynamic(aero_builder=aero_builder))
        self.mphys_add_scenario('cruise_higher_aoa',ScenarioAerodynamic(aero_builder=aero_builder))

        for dv in ['q_inf', 'vel', 'nu']:
            self.connect(dv, f'cruise.{dv}')
            self.connect(dv, f'cruise_higher_aoa.{dv}')
        for dv in ['aoa', 'mach']:
            self.connect(f'{dv}0', f'cruise.{dv}')
            self.connect(f'{dv}1', f'cruise_higher_aoa.{dv}')

        self.connect('mesh.x_aero0',['cruise.x_aero','cruise_higher_aoa.x_aero'])

prob = om.Problem()
prob.model = Top()
prob.setup()

om.n2(prob, show_browser=False, outfile='vlm_aero_2cruises.html')

prob.run_model()
if MPI.COMM_WORLD.rank == 0:
    for scenario in ['cruise','cruise_higher_aoa']:
        print('%s: C_L = %f, C_D = %f' % (scenario, prob['%s.C_L'%scenario], prob['%s.C_D'%scenario]))
