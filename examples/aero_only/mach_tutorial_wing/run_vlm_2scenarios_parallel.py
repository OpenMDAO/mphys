from mpi4py import MPI
import openmdao.api as om

from mphys import MultipointParallel
from mphys.mphys_vlm import VlmBuilderAeroOnly
from mphys.scenario_aero import ScenarioAero

class ParallelCruises(MultipointParallel):
    def setup(self):
        # VLM options
        mesh_file = 'wing_VLM.dat'

        aero_builder = VlmBuilderAeroOnly(mesh_file)
        self.mphys_add_scenario('cruise',ScenarioAero(aero_builder=aero_builder,
                                                      in_MultipointParallel=True))

        self.mphys_add_scenario('cruise_higher_aoa',ScenarioAero(aero_builder=aero_builder,
                                                                 in_MultipointParallel=True))

class Top(om.Group):
    def setup(self):
        mach0 = 0.85,
        mach1 = 0.80,
        aoa0 = 0.0
        aoa1 = 1.0
        q_inf = 3000.
        vel = 178.
        mu = 3.5E-5

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa0', val=aoa0, units='deg')
        dvs.add_output('aoa1', val=aoa1, units='deg')
        dvs.add_output('mach0', mach0)
        dvs.add_output('mach1', mach1)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('mu', mu)

        self.add_subsystem('mp',ParallelCruises())
        for dv in ['q_inf', 'vel', 'mu']:
            self.connect(dv, f'mp.cruise.{dv}')
            self.connect(dv, f'mp.cruise_higher_aoa.{dv}')
        for dv in ['mach', 'aoa']:
            self.connect(f'{dv}0',f'mp.cruise.{dv}')
            self.connect(f'{dv}1',f'mp.cruise_higher_aoa.{dv}')

prob = om.Problem()
prob.model = Top()
prob.setup()

om.n2(prob, show_browser=False, outfile='vlm_aero_2cruises_parallel.html')

prob.run_model()

for scenario in ['cruise','cruise_higher_aoa']:
    cl = prob.get_val(f'mp.{scenario}.C_L',get_remote=True)
    cd = prob.get_val(f'mp.{scenario}.C_D',get_remote=True)
    if MPI.COMM_WORLD.rank == 0:
        print(f'{scenario}: C_L = {cl}, C_D={cd}')
