import numpy as np
from mpi4py import MPI

import openmdao.api as om
from mphys.multipoint import Multipoint
from mphys.scenario_aero import ScenarioAero
from mphys.mphys_vlm import VlmBuilder

comm = MPI.COMM_WORLD


class Top(Multipoint):
    def setup(self):
        mesh_file = '../input_files/debug_VLM.dat'
        mach = 0.85,
        aoa = 1.0
        q_inf = 25000.
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

        self.add_subsystem('mesh', aero_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('cruise', ScenarioAero(aero_builder=aero_builder))
        self.connect('mesh.x_aero0', 'cruise.x_aero')

        for dv in ['aoa', 'mach', 'q_inf', 'vel', 'mu']:
            self.connect(dv, f'cruise.{dv}')


prob = om.Problem()
prob.model = Top()
prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs', compact_print=True)
