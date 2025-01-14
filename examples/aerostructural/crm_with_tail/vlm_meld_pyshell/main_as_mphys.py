import numpy as np
import openmdao.api as om
# MELD builder from funtofem
from funtofem.mphys import MeldBuilder
from mappings import dv_map, f_struct_map
from mpi4py import MPI
from pyshell.mphys import PyshellBuilder, PyshellInertialForce
# case-specific components
from util_components import GeometryBuilder, LumpDvs
# in-house aero and structural solvers
from vlm_solver.mphys_vlm import VlmBuilder

from mphys import Multipoint
from mphys.scenarios.aerostructural import ScenarioAeroStructural

comm = MPI.COMM_WORLD
rank = comm.rank

struct_nprocs = min([comm.Get_size(), 15]) # number of procs to run the structural solver on

# different multi-body options to test
body_option = 5 # 1=flexible wing only
                # 2=flexible wing, rigid tail
                # 3=coupled wing, inertia-only tail
                # 4=inertia-only wing, coupled tail
                # 5=flexible wing and tail

if body_option==1:
    struct_mesh_file = 'CRM_box_2nd.bdf'
    aero_mesh_file = 'CRM_VLM_mesh_extended_coarse.dat'
    body_tags_meld = [] # will couple all aero/struct nodes by default
    body_tags_geom = [{'aero': [0], 'struct': np.arange(240)}]
    dv_struct = 0.01*np.ones(240)
    meld_n = 200
    meld_beta = 0.5
elif body_option==2:
    struct_mesh_file = 'CRM_box_2nd.bdf'
    aero_mesh_file = 'CRM_VLM_mesh_extended_coarse_tail.dat'
    body_tags_meld = [{'aero': [0], 'struct': np.arange(240)}] # couple wing VLM boundary (0) with wing property groups (0-239)
    body_tags_geom = [{'aero': [0], 'struct': np.arange(240)},
                      {'aero': [1], 'struct': []}] # move aero-only tail with geometry builder
    dv_struct = 0.01*np.ones(240)
    meld_n = 200
    meld_beta = 0.5
elif body_option==3:
    struct_mesh_file = 'CRM_box_2nd_tailbox.bdf'
    aero_mesh_file = 'CRM_VLM_mesh_extended_coarse.dat'
    body_tags_meld = [{'aero': [0], 'struct': np.arange(240)}]
    body_tags_geom = [{'aero': [0], 'struct': np.arange(240)},
                      {'aero': [],  'struct': np.arange(240,285)}] # move struct-only tail with geometry builder
    dv_struct = 0.01*np.ones(285)
    dv_struct[-45:] = 0.002 # thinner tail, for exaggerated deflection
    meld_n = 200
    meld_beta = 0.5
elif body_option==4:
    struct_mesh_file = 'CRM_box_2nd_tailbox.bdf'
    aero_mesh_file = 'CRM_VLM_tail.dat'
    body_tags_meld = [{'aero': [0], 'struct': np.arange(240,285)}]
    body_tags_geom = [{'aero': [],  'struct': np.arange(240)},
                      {'aero': [0], 'struct': np.arange(240,285)}]
    dv_struct = 0.01*np.ones(285)
    dv_struct[-45:] = 0.002
    meld_n = 1000
    meld_beta = 0.5
elif body_option==5:
    struct_mesh_file = 'CRM_box_2nd_tailbox.bdf'
    aero_mesh_file = 'CRM_VLM_mesh_extended_coarse_tail.dat'
    body_tags_meld = [{'aero': [0], 'struct': np.arange(240)},
                      {'aero': [1], 'struct': np.arange(240,285)}]
    body_tags_geom = [{'aero': [0], 'struct': np.arange(240)},
                      {'aero': [1], 'struct': np.arange(240,285)}]
    dv_struct = 0.01*np.ones(285)
    dv_struct[-45:] = 0.002
    meld_n = [200,1000]
    meld_beta = 0.5

# shape dvs (separate sweep and span for wing and tail)
span = 29.39
sweep_dv = [0.]*len(body_tags_geom)
span_dv = [0.]*len(body_tags_geom)

# aero conditions
mach = 0.85
q_inf = 12930.
vel = 254.
nu = 3.5E-5
aoa = 3.
load_factor = 2.5

## Mphys
class Top(Multipoint):
    def setup(self):

        ## pyshell setup
        pyshell_setup = {'mesh_file'    : struct_mesh_file,
                         'dv_map'       : dv_map,
                         'f_struct_map' : f_struct_map,
                         'load_list'    : ['f_inertial', 'f_struct'],}

        if body_option>2: # two separate KS functions for wing and tail, based on element number
            pyshell_setup['KS_divisions'] = [np.arange(10584), np.arange(10584,13664)]

        struct_builder = PyshellBuilder(pyshell_setup, num_procs=struct_nprocs)
        struct_builder.initialize(self.comm)

        self.add_subsystem('struct_mesh',struct_builder.get_mesh_coordinate_subsystem())

        ## structural design variables
        self.add_subsystem('thickness_dv', om.IndepVarComp(), promotes=['*'])
        self.add_subsystem('thickness_lumping',LumpDvs(
            N_dv=len(dv_struct),
            dv=dv_struct)
            )
        self.thickness_dv.add_output('thickness_dv',1.)
        self.connect('thickness_dv','thickness_lumping.lumped_dv')

        ## ivc
        self.add_subsystem('ivc_params', om.IndepVarComp(), promotes=['*'])
        self.ivc_params.add_output('mach', val=mach)
        self.ivc_params.add_output('q_inf', val=q_inf)
        self.ivc_params.add_output('vel', val=vel)
        self.ivc_params.add_output('nu', val=nu)
        self.ivc_params.add_output('load_factor', val=load_factor)
        self.ivc_params.add_output('sweep_dv', val=sweep_dv)
        self.ivc_params.add_output('span_dv', val=span_dv)
        self.ivc_params.add_output('aoa', val=aoa, units='deg')

        ## VLM builder
        aero_builder = VlmBuilder(aero_mesh_file, CD_misc=0.008)
        aero_builder.initialize(comm)

        self.add_subsystem('aero_mesh',aero_builder.get_mesh_coordinate_subsystem())

        ## geometry builder
        geom_builder = GeometryBuilder({'aero': aero_builder, 'struct': struct_builder},
                                       span=span,body_tags=body_tags_geom)
        self.add_subsystem('geometry', geom_builder.get_mesh_coordinate_subsystem(), promotes=['*'])

        self.connect('struct_mesh.x_struct0','x_struct_in')
        self.connect('aero_mesh.x_aero0','x_aero_in')

        ## inertial load
        self.add_subsystem('inertial_loads',PyshellInertialForce(gravity=9.81, struct_solver=struct_builder.solver))
        self.connect('thickness_lumping.dv_struct', 'inertial_loads.dv_struct')
        self.connect('x_struct0', 'inertial_loads.x_struct0')
        self.connect('load_factor','inertial_loads.load_factor')

        ## meld builder
        xfer_builder = MeldBuilder(aero_builder, struct_builder, isym=1, n=meld_n, beta=meld_beta, body_tags=body_tags_meld)
        xfer_builder.initialize(comm)

        ## add scenario
        nonlinear_solver = om.NonlinearBlockGS(
            maxiter=20, iprint=2, use_aitken=True, aitken_initial_factor=0.5)
        linear_solver = om.LinearBlockGS(
            maxiter=20, iprint=2, use_aitken=True, aitken_initial_factor=0.5)
        self.mphys_add_scenario('pullup',ScenarioAeroStructural(
                                            aero_builder=aero_builder,
                                            struct_builder=struct_builder,
                                            ldxfer_builder=xfer_builder),
                                        coupling_nonlinear_solver=nonlinear_solver,
                                        coupling_linear_solver=linear_solver)

        for var in ['mach', 'vel', 'nu', 'aoa', 'q_inf', 'x_aero0', 'x_struct0']:
            self.connect(var, 'pullup.'+var)
        self.connect('thickness_lumping.dv_struct', 'pullup.dv_struct')
        self.connect('inertial_loads.f_inertial', 'pullup.f_inertial')

prob = om.Problem()
prob.model = Top()

prob.setup(mode='rev',force_alloc_complex=False)
prob.run_model()
om.n2(prob, show_browser=False, outfile='n2.html')

if rank==0:
    print('CD = ',prob['pullup.C_D'])
    print('CL = ',prob['pullup.C_L'])
    print('KS = ',prob['pullup.KS_stress'])

## check totals
prob.check_totals(
            of=['pullup.mass','pullup.KS_stress','pullup.C_D','pullup.C_L'],
            wrt=['thickness_dv','aoa','sweep_dv','span_dv'],
            compact_print=True,
            directional=True)
