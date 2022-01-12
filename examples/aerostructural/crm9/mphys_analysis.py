#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.mphys_fun3d import Fun3dSfeBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_meld import MeldBuilder
from mphys.mphys_vlm import VlmBuilder

import tacs_setup
from structural_patches_component import LumpPatches

from tacs import functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

use_fun3d = True

class Top(Multipoint):
    def setup(self):
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        aoa = 0.0
        mach = 0.85
        q_inf = 120.0
        vel = 217.6
        nu = 1.4E-5

        if use_fun3d:
            # FUN3D options
            boundary_tag_list = [3]
            aero_builder = Fun3dSfeBuilder(boundary_tag_list, input_file='input.cfg')
            aero_builder.initialize(self.comm)

            dvs.add_output('aoa', val=aoa, units='deg')
            dvs.add_output('mach', val=mach)
            dvs.add_output('reynolds', val=600000.0)
            dvs.add_output('q_inf', val=q_inf)
            dvs.add_output('yaw', val=0.0, units='deg')
            aero_dvs = ['aoa','mach','reynolds','q_inf','yaw']
        else:
            mesh_file = 'CRM_VLM_mesh_extended.dat'
            aero_builder = VlmBuilder(mesh_file)
            aero_builder.initialize(self.comm)

            dvs.add_output('aoa', val=aoa, units='deg')
            dvs.add_output('mach', mach)
            dvs.add_output('q_inf', q_inf)
            dvs.add_output('vel', vel)
            dvs.add_output('nu', nu)
            aero_dvs = ['aoa','mach','q_inf','vel','nu']

        # TACS options
        tacs_options = {
            'element_callback': tacs_setup.element_callback,
            'mesh_file'   : 'CRM_box_2nd.bdf'
        }
        struct_builder = TacsBuilder(tacs_options, coupled=True)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        dvs.add_output('thickness_lumped', val=0.01)
        self.add_subsystem('lumper',LumpPatches(N=ndv_struct))
        self.connect('thickness_lumped','lumper.thickness_lumped')

        # Transfer scheme options
        isym = 1
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)
        ldxfer_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero',aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem())

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8)
        self.mphys_add_scenario('cruise',ScenarioAeroStructural(aero_builder=aero_builder,
                                                                struct_builder=struct_builder,
                                                                ldxfer_builder=ldxfer_builder),
                                         nonlinear_solver, linear_solver)

        for discipline in ['aero','struct']:
            self.mphys_connect_scenario_coordinate_source('mesh_%s' % discipline, 'cruise', discipline)

        self.connect('lumper.thickness', ['cruise.dv_struct'])
        for dv in aero_dvs:
            self.connect(dv, f'cruise.{dv}')

    def configure(self):
        # create the tacs problems for adding evalfuncs and fixed structural loads to the analysis point.
        # This is custom to the tacs based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the struct group.
        fea_assembler = self.cruise.coupling.struct.fea_assembler

        # ==============================================================================
        # Setup structural problem
        # ==============================================================================
        # Structural problem
        sp = fea_assembler.createStaticProblem(name='cruise')
        # Add TACS Functions
        compIDs = fea_assembler.selectCompIDs(nGroup=-1)
        sp.addFunction('mass', functions.StructuralMass, compIDs=compIDs)
        sp.addFunction('ks_vmfailure', functions.KSFailure, ksWeight=50.0, safetyFactor=1.0)

        # Add gravity load
        g = np.array([0.0, 0.0, -9.81])  # m/s^2
        sp.addInertialLoad(g)

        # here we set the tacs problems for every load case we have.
        self.cruise.coupling.struct.mphys_set_sp(sp)
        self.cruise.struct_post.mphys_set_sp(sp)


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup(mode='rev')
prob.final_setup()
om.n2(prob, show_browser=False, outfile='crm_aerostruct.html')
prob.run_model()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('C_L =',prob['cruise.C_L'])
    print('C_D =',prob['cruise.C_D'])
    print('KS =',prob['cruise.struct_post.ks_vmfailure'])
#output = prob.check_totals(of=['mp_group.s0.aero_funcs.Lift'], wrt=['thickness_lumped'],)
#if MPI.COMM_WORLD.rank == 0:
#    print('check_totals output',output)

