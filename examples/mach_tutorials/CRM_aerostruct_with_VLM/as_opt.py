#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from tacs import elements, constitutive, functions, TACS

from mphys.mphys_multipoint import MPHYS_Multipoint
from mphys.mphys_vlm import VLM_builder
from mphys.mphys_tacs import TACS_builder
from mphys.mphys_modal_solver import ModalBuilder
from mphys.mphys_meld import MELD_builder

from structural_patches_component import PatchList, DesignPatches, PatchSmoothness, LumpPatches
from wing_geometry_component import WingGeometry

patches = PatchList('CRM_box_2nd.bdf')
patches.read_families()
patches.create_DVs()

strutural_patch_lumping = False    # can you add this, and the patches, into the setup of Top?

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
        
        # lump all structural dvs attached to a given component (upper skin, e.g.) into a single value: useful for checking_totals

        if strutural_patch_lumping is True:
            self.add_subsystem('upper_skin_lumper',LumpPatches(N=patches.n_us)) 
            self.add_subsystem('lower_skin_lumper',LumpPatches(N=patches.n_ls))  
            self.add_subsystem('le_spar_lumper',LumpPatches(N=patches.n_le))  
            self.add_subsystem('te_spar_lumper',LumpPatches(N=patches.n_te))  
            self.add_subsystem('rib_lumper',LumpPatches(N=patches.n_rib))     

        # structural mapper: map structural component arrays into the unified array that goes into tacs

        self.add_subsystem('struct_mapper',DesignPatches(patch_list=patches))

        # patch smoothness constraints

        self.add_subsystem('upper_skin_smoothness',PatchSmoothness(N=patches.n_us))
        self.add_subsystem('lower_skin_smoothness',PatchSmoothness(N=patches.n_ls))
        self.add_subsystem('le_spar_smoothness',PatchSmoothness(N=patches.n_le))
        self.add_subsystem('te_spar_smoothness',PatchSmoothness(N=patches.n_te))
        
        # geometry mapper

        struct_builder.init_solver(MPI.COMM_WORLD)
        tacs_solver = struct_builder.get_solver()
        xpts = tacs_solver.createNodeVec()
        tacs_solver.getNodes(xpts)
        x_s0 = xpts.getArray()

        x_a0 = vlm_builder.options['x_a0']

        self.add_subsystem('geometry_mapper',WingGeometry(xs=x_s0, xa=x_a0))


        f=open("prop_ID.dat","w+")
        for ielem, elem in enumerate(tacs_solver.getElements()):
            f.write(str(elem.getComponentNum()) + '\n')
        f.close()


        
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

        if strutural_patch_lumping is False:
            self.dvs.add_output('upper_skin_thickness',     val=initial_thickness, shape = patches.n_us)
            self.dvs.add_output('lower_skin_thickness',     val=initial_thickness, shape = patches.n_ls)
            self.dvs.add_output('le_spar_thickness',        val=initial_thickness, shape = patches.n_le)
            self.dvs.add_output('te_spar_thickness',        val=initial_thickness, shape = patches.n_te)
            self.dvs.add_output('rib_thickness',            val=initial_thickness, shape = patches.n_rib)

            self.connect('upper_skin_thickness','struct_mapper.upper_skin_thickness')
            self.connect('lower_skin_thickness','struct_mapper.lower_skin_thickness')
            self.connect('le_spar_thickness','struct_mapper.le_spar_thickness')
            self.connect('te_spar_thickness','struct_mapper.te_spar_thickness')
            self.connect('rib_thickness','struct_mapper.rib_thickness')

        else:
            self.dvs.add_output('upper_skin_thickness_lumped',     val=initial_thickness, shape = 1)
            self.dvs.add_output('lower_skin_thickness_lumped',     val=initial_thickness, shape = 1)
            self.dvs.add_output('le_spar_thickness_lumped',        val=initial_thickness, shape = 1)
            self.dvs.add_output('te_spar_thickness_lumped',        val=initial_thickness, shape = 1)
            self.dvs.add_output('rib_thickness_lumped',            val=initial_thickness, shape = 1)

            self.connect('upper_skin_thickness_lumped','upper_skin_lumper.thickness_lumped')
            self.connect('lower_skin_thickness_lumped','lower_skin_lumper.thickness_lumped')
            self.connect('le_spar_thickness_lumped','le_spar_lumper.thickness_lumped')
            self.connect('te_spar_thickness_lumped','te_spar_lumper.thickness_lumped')
            self.connect('rib_thickness_lumped','rib_lumper.thickness_lumped')

            self.connect('upper_skin_lumper.thickness','struct_mapper.upper_skin_thickness')
            self.connect('lower_skin_lumper.thickness','struct_mapper.lower_skin_thickness')
            self.connect('le_spar_lumper.thickness','struct_mapper.le_spar_thickness')
            self.connect('te_spar_lumper.thickness','struct_mapper.te_spar_thickness')
            self.connect('rib_lumper.thickness','struct_mapper.rib_thickness')

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

        if strutural_patch_lumping is False:
            self.connect('upper_skin_thickness','upper_skin_smoothness.thickness')
            self.connect('lower_skin_thickness','lower_skin_smoothness.thickness')
            self.connect('le_spar_thickness','le_spar_smoothness.thickness')
            self.connect('te_spar_thickness','te_spar_smoothness.thickness')
        else:
            self.connect('upper_skin_lumper.thickness','upper_skin_smoothness.thickness')
            self.connect('lower_skin_lumper.thickness','lower_skin_smoothness.thickness')
            self.connect('le_spar_lumper.thickness','le_spar_smoothness.thickness')
            self.connect('te_spar_lumper.thickness','te_spar_smoothness.thickness')

        # connect solver data

        self.connect('struct_mapper.dv_struct', ['mp_group.s0.struct.dv_struct'])

        # connect the geometry mesh outputs

        points = self.mp_group.mphys_add_coordinate_input()
        self.connect('geometry_mapper.x_s0_mesh','mp_group.struct_points')
        self.connect('geometry_mapper.x_a0_mesh','mp_group.aero_points')

################################################################################
# OpenMDAO setup
################################################################################

prob = om.Problem()
prob.model = Top()
model = prob.model

model.nonlinear_solver = om.NonlinearRunOnce()
model.linear_solver = om.LinearRunOnce()

prob.setup(mode='rev',force_alloc_complex=True)
om.n2(prob, show_browser=False, outfile='CRM_mphys_as_vlm.html')

model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=False, rtol = 1E-9, atol=1E-10)
model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-9, atol=1e-10)

prob.run_model()

#prob.check_totals(of=['mp_group.s0.aero.forces.CD', 'mp_group.s0.struct.mass.mass', 'mp_group.s0.struct.funcs.f_struct'], wrt=['alpha', 'span_delta'], method='cs')

prob.check_totals(of=['mp_group.s0.aero.forces.CD', 'mp_group.s0.struct.mass.mass', 'mp_group.s0.struct.funcs.f_struct'], wrt=['alpha', 'upper_skin_thickness_lumped', 'lower_skin_thickness_lumped', 'le_spar_thickness_lumped', 'te_spar_thickness_lumped', 'rib_thickness_lumped', 'root_chord_delta', 'tip_chord_delta', 'tip_sweep_delta', 'span_delta', 'wing_thickness_delta', 'wing_twist_delta'], method='cs')

# push new structural changes, and pull to mac
# fuel volume, and fuel load
# planform area: comes from geometry tool?
# fuel burn
# multiple load cases
# spar depth
# inertial load
# trim
# wing thickness UB and LB
# some way to get # of thickness and twist variabels non-hard-coded
# fuel DV, and fuel matching.  reserve fuel?
# LGW vs FB, and weigting between the two
# buffet

# force vector does not include load factor, so you'll need to include it when you sum aero + inertial + fuel


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
