#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from tacs import elements, constitutive, functions, TACS

from mphys.mphys_multipoint_hacked import MPHYS_Multipoint
from mphys.mphys_vlm import VLM_builder
from mphys.mphys_tacs import TACS_builder
from mphys.mphys_modal_solver import ModalBuilder
from mphys.mphys_meld import MELD_builder

from structural_patches_component import DesignPatches, PatchSmoothness
from wing_geometry_component import WingGeometry

class Top(om.Group):

    def setup(self):
        # VLM options
        aero_options = {
            'mesh_file':'CRM_VLM_mesh_extended.dat',
            'mach':0.85,
            'alpha':0*np.pi/180.,
            'q_inf':9000.,
            'vel':178.,
            'mu':3.5E-5,
        }

        # VLM mesh read

        def read_VLM_mesh(mesh):
            f=open(mesh, "r")
            contents = f.read().split()

            a = [i for i in contents if 'NODES' in i][0]
            N_nodes = int(a[a.find("=")+1:a.find(",")])
            a = [i for i in contents if 'ELEMENTS' in i][0]
            N_elements = int(a[a.find("=")+1:a.find(",")])
	    
            start = len(contents)-N_nodes*3-N_elements*4
            a = np.array(contents[start:start+N_nodes*3],'float')
            X = a[0:N_nodes*3:3]
            Y = a[1:N_nodes*3:3]
            Z = a[2:N_nodes*3:3]
            a = np.array(contents[start+N_nodes*3:None],'int')
            quad = np.reshape(a,[N_elements,4])

            xa = np.c_[X,Y,Z].flatten(order='C')

            f.close()

            return N_nodes, N_elements, xa, quad

        aero_options['N_nodes'], aero_options['N_elements'], aero_options['x_a0'], aero_options['quad'] = read_VLM_mesh(aero_options['mesh_file'])

        # VLM builder
        vlm_builder = VLM_builder(aero_options)

        # TACS setup

        def add_elements(mesh):
            rho = 2780.0            # density, kg/m^3
            E = 73.1e9              # elastic modulus, Pa
            nu = 0.33               # poisson's ratio
            kcorr = 5.0 / 6.0       # shear correction factor
            ys = 324.0e6            # yield stress, Pa
            thickness= 0.003
            min_thickness = 0.002
            max_thickness = 0.05

            num_components = mesh.getNumComponents()
            for i in range(num_components):
                descript = mesh.getElementDescript(i)
                stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                            min_thickness, max_thickness)
                element = None
                if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
                    element = elements.MITCShell(2,stiff,component_num=i)
                mesh.setElement(i, element)

            ndof = 6
            ndv = num_components

            return ndof, ndv

        def get_funcs(tacs):
            ks_weight = 50.0
            return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

        def f5_writer(tacs):
            flag = (TACS.ToFH5.NODES |
                    TACS.ToFH5.DISPLACEMENTS |
                    TACS.ToFH5.STRAINS |
                    TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile('wingbox.f5')


        # common setup options
        tacs_setup = {'add_elements': add_elements,
                    'get_funcs'   : get_funcs,
                    'mesh_file'   : 'CRM_box_2nd.bdf',
                    'f5_writer'   : f5_writer }

        struct_builder = TACS_builder(tacs_setup)

        # MELD setup
        meld_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}

        # MELD builder
        meld_builder = MELD_builder(meld_options, vlm_builder, struct_builder)

        ################################################################################
        # MPHYS setup
        ################################################################################
        # ivc to keep the top level DVs
        self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        self.add_subsystem('struct_mapper',DesignPatches(bdf='CRM_box_2nd.bdf'), promotes=['*'])

        self.add_subsystem('upper_skin_smoothness',PatchSmoothness(N=48))
        self.add_subsystem('lower_skin_smoothness',PatchSmoothness(N=48))
        self.add_subsystem('le_spar_smoothness',PatchSmoothness(N=48))
        self.add_subsystem('te_spar_smoothness',PatchSmoothness(N=44))

        def read_BDF_mesh(mesh):
            f=open(mesh, "r")
            contents = f.read().split()
            a = [i for i, s in enumerate(contents) if 'GRID*' in s]
            GRID = np.zeros([len(a),3],'float')

            for i in range(0,len(a)):
                GRID[i,:] = np.array([float(contents[a[i]+3]),float(contents[a[i]+4][0:-1]),float(contents[a[i]+8])])

            GRID = GRID.flatten(order='C')            
            f.close()
            return GRID

        x_s0 = read_BDF_mesh('CRM_box_2nd.bdf')
        x_a0 = vlm_builder.options['x_a0']
        self.add_subsystem('geometry_mapper',WingGeometry(xs=x_s0, xa=x_a0))

        # each AS_Multipoint instance can keep multiple points with the same formulation
        mp = self.add_subsystem(
            'mp_group',
            MPHYS_Multipoint(
                aero_builder   = vlm_builder,
                struct_builder = struct_builder,
                xfer_builder   = meld_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

    def configure(self):

        # add AoA DV

        self.dvs.add_output('alpha', val=0*np.pi/180.)
        self.connect('alpha', 'mp_group.s0.aero.alpha')

        # add the structural thickness DVs
        initial_thickness = 0.01
        self.dvs.add_output('upper_skin_thickness',     val=initial_thickness, shape = 48)
        self.dvs.add_output('lower_skin_thickness',     val=initial_thickness, shape = 48)
        self.dvs.add_output('le_spar_thickness',        val=initial_thickness, shape = 48)
        self.dvs.add_output('te_spar_thickness',        val=initial_thickness, shape = 44)
        self.dvs.add_output('rib_thickness',            val=initial_thickness, shape = 49)

        # add the geometry DVS
        self.dvs.add_output('root_chord_delta',         val=0.0)
        self.dvs.add_output('tip_chord_delta',          val=0.0)
        self.dvs.add_output('tip_sweep_delta',          val=0.0)
        self.dvs.add_output('span_delta',               val=0.0)
        self.dvs.add_output('wing_thickness_delta',     val=0.0,               shape = 6)
        self.dvs.add_output('wing_twist_delta',         val=0.0,               shape = 5)

        ## connect the geometry DVs
        self.connect('root_chord_delta','geometry_mapper.root_chord_delta')
        self.connect('tip_chord_delta','geometry_mapper.tip_chord_delta')
        self.connect('tip_sweep_delta','geometry_mapper.tip_sweep_delta')
        self.connect('span_delta','geometry_mapper.span_delta')
        self.connect('wing_thickness_delta','geometry_mapper.wing_thickness_delta')
        self.connect('wing_twist_delta','geometry_mapper.wing_twist_delta')

        # connect the smoothness constraints
        self.connect('upper_skin_thickness','upper_skin_smoothness.thickness')
        self.connect('lower_skin_thickness','lower_skin_smoothness.thickness')
        self.connect('le_spar_thickness','le_spar_smoothness.thickness')
        self.connect('te_spar_thickness','te_spar_smoothness.thickness')

        # connect solver data
        self.connect('dv_struct', ['mp_group.s0.struct.dv_struct'])

        # make the connections that you aren't letting MPHYS_Multipoint do anymore
        self.connect('geometry_mapper.x_s0_mesh','mp_group.s0.struct.x_s0')
        self.connect('geometry_mapper.x_s0_mesh','mp_group.s0.disp_xfer.x_s0')
        self.connect('geometry_mapper.x_s0_mesh','mp_group.s0.load_xfer.x_s0')
        self.connect('geometry_mapper.x_a0_mesh','mp_group.s0.aero.x_a0')
        self.connect('geometry_mapper.x_a0_mesh','mp_group.s0.disp_xfer.x_a0')
        self.connect('geometry_mapper.x_a0_mesh','mp_group.s0.load_xfer.x_a0')


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model

# optimization set up
#prob.model.add_design_var('alpha',lower=-5*np.pi/180, upper=10*np.pi/180.0, ref=1.0)
#prob.model.add_design_var('ribs',        lower=0.003, upper=0.020, ref=0.005)
#prob.model.add_design_var('le_spar',     lower=0.003, upper=0.020, ref=0.005)
#prob.model.add_design_var('te_spar',     lower=0.003, upper=0.020, ref=0.005)
#prob.model.add_design_var('up_skin',     lower=0.003, upper=0.020, ref=0.005)
#prob.model.add_design_var('lo_skin',     lower=0.003, upper=0.020, ref=0.005)
#prob.model.add_design_var('up_stringer', lower=0.003, upper=0.020, ref=0.005)
#prob.model.add_design_var('lo_stringer', lower=0.003, upper=0.020, ref=0.005)

#prob.model.add_objective('mp_group.s0.struct.mass.mass',ref=1000.0)
#prob.model.add_constraint('mp_group.s0.aero.forces.CL',ref=1.0,equals=0.5)
#prob.model.add_constraint('mp_group.s0.struct.funcs.f_struct',ref=1.0, upper = 2.0/3.0)

#prob.model.add_constraint('le_spar_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)
#prob.model.add_constraint('te_spar_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)
#prob.model.add_constraint('up_skin_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)
#prob.model.add_constraint('lo_skin_smoothness.diff', ref=1e-3, upper = 0.0, linear=True)

# optional but we can set it here.
model.nonlinear_solver = om.NonlinearRunOnce()
model.linear_solver = om.LinearRunOnce()

##prob.driver = om.ScipyOptimizeDriver(debug_print=['ln_cons','nl_cons','objs','totals'])
#prob.driver = om.ScipyOptimizeDriver()
#prob.driver.options['optimizer'] = 'SLSQP'
#prob.driver.options['tol'] = 1e-3
#prob.driver.options['disp'] = True

#prob.driver.recording_options['includes'] = ['*']
#prob.driver.recording_options['record_objectives'] = True
#prob.driver.recording_options['record_constraints'] = True
#prob.driver.recording_options['record_desvars'] = True

#recorder = om.SqliteRecorder("cases.sql")
#prob.driver.add_recorder(recorder)

prob.setup()

#model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=20, iprint=2, use_aitken=True, rtol = 1E-7, atol=1E-8)
#model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=20, iprint=2, rtol = 1e-7, atol=1e-8)

om.n2(prob, show_browser=False, outfile='CRM_mphys_as_vlm.html')
#prob.run_driver()

#cr = om.CaseReader("cases.sql")
#driver_cases = cr.list_cases('driver')

#matrix = np.zeros((len(driver_cases),4))
#for i, case_id in enumerate(driver_cases):
#    matrix[i,0] = i
#    case = cr.get_case(case_id)
#    matrix[i,1] = case.get_objectives()['mp_group.s0.struct.mass.mass'][0]
#    matrix[i,2] = case.get_constraints()['mp_group.s0.aero.forces.CL'][0]
#    matrix[i,3] = case.get_constraints()['mp_group.s0.struct.funcs.f_struct'][0]
#np.savetxt('history.dat',matrix)
