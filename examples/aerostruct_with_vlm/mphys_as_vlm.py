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

class Top(Multipoint):

    def setup(self):
        self.modal_struct = False

        # VLM
        mesh_file = 'wing_VLM.dat'
        mach = 0.85,
        aoa = 2.0
        q_inf = 3000.
        vel = 178.
        mu = 3.5E-5

        aero_builder = VlmBuilder(mesh_file)
        aero_builder.initialize(self.comm)

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('mu', mu)

        self.add_subsystem('mesh_aero',aero_builder.get_mesh_coordinate_subsystem())

        # TACS
        tacs_options = {'add_elements': tacs_setup.add_elements,
                        'get_funcs'   : tacs_setup.get_funcs,
                        'mesh_file'   : 'wingbox_Y_Z_flip.bdf',
                        'f5_writer'   : tacs_setup.f5_writer }

        struct_builder = TacsBuilder(tacs_options)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem())

        dvs.add_output('dv_struct', np.array(ndv_struct*[0.002]))

        # MELD setup
        isym = 1
        xfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)
        xfer_builder.initialize(self.comm)

        nonlinear_solver = om.NonlinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol = 1E-14, atol=1E-14)
        linear_solver = om.LinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol = 1e-14, atol=1e-14)

        self.mphys_add_scenario('cruise',ScenarioAeroStructural(aero_builder=aero_builder,
                                                                struct_builder=struct_builder,
                                                                xfer_builder=xfer_builder),
                                         coupling_nonlinear_solver=nonlinear_solver,
                                         coupling_linear_solver=linear_solver)


        for discipline in ['aero','struct']:
            self.mphys_connect_scenario_coordinate_source('mesh_%s' % discipline ,'cruise', discipline)

        for dv in ['aoa','q_inf','vel','mu','mach', 'dv_struct']:
            self.connect(dv, 'cruise.%s' % dv)

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model

# optional but we can set it here.
model.nonlinear_solver = om.NonlinearRunOnce()
model.linear_solver = om.LinearRunOnce()


prob.setup()

om.n2(prob, show_browser=False, outfile='mphys_as_vlm.html')

prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print('func_struct =',prob['cruise.func_struct'])
    print('mass =',prob['cruise.mass'])
    print('C_L =',prob['cruise.C_L'])
