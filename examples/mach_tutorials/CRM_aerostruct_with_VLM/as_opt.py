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

y_knot = np.array([0,3.0999,10.979510169492,16.968333898305,23.456226271186,29.44505])
LE_knot = np.array([23.06635,25.251679291894,31.205055607314,35.736770145885000,40.641208661042000,45.16478])
TE_knot = np.array([36.527275,37.057127604678,38.428622653948,41.498242368371000,44.820820782553000,47.887096577690000])

strutural_patch_lumping = False    

N_mp = 2
aero_parameters = {
    'mach': [0.85, .64],
    'q_inf': [12930., 28800.],
    'vel': [254., 217.6],
    'mu': [3.5E-5, 1.4E-5],
    'alpha': np.array([1., 4.])*np.pi/180.,
}

trim_parameters = {
    'load_factor': [1.0, 2.5],
    'load_case_fuel_burned': [.5, 1.0],
}


# need to add some of this to an input deck, which would go in setup?
# patches maybe could go in setup too?





class Top(om.Group):

    def setup(self):

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

        aero_options = {}
        aero_options['N_nodes'], aero_options['N_elements'], aero_options['x_a0'], aero_options['quad'] = read_VLM_mesh('CRM_VLM_mesh_extended.dat')

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

        # ivc for the multipoint parameters

        self.add_subsystem('mp_parameters', om.IndepVarComp(), promotes=['*'])

        # ivc to keep the top level DVs

        self.add_subsystem('sizing_dvs', om.IndepVarComp(), promotes=['*'])
        self.add_subsystem('geometric_dvs', om.IndepVarComp(), promotes=['*'])
        self.add_subsystem('fuel_dvs', om.IndepVarComp(), promotes=['*'])
        self.add_subsystem('trim_dvs', om.IndepVarComp(), promotes=['*'])
        
        # lump all structural dvs attached to a given component (upper skin, e.g.) into a single value: useful for checking_totals

        if strutural_patch_lumping is True:
            self.add_subsystem('struct_lumping',om.Group())
            for comp, n in zip(['upper_skin','lower_skin','le_spar','te_spar','rib'],[patches.n_us,patches.n_ls,patches.n_le,patches.n_te,patches.n_rib]):
                self.struct_lumping.add_subsystem(comp+'_lumper',LumpPatches(N=n)) 

        # structural mapper: map structural component arrays into the unified array that goes into tacs

        self.add_subsystem('struct_mapper',DesignPatches(patch_list=patches))

        # patch smoothness constraints

        self.add_subsystem('struct_smoothness',om.Group())
        for comp, n in zip(['upper_skin','lower_skin','le_spar','te_spar'],[patches.n_us,patches.n_ls,patches.n_le,patches.n_te]):        
            self.struct_smoothness.add_subsystem(comp+'_smoothness',PatchSmoothness(N=n))
        
        # geometry mapper

        struct_builder.init_solver(MPI.COMM_WORLD)
        tacs_solver = struct_builder.get_solver()
        xpts = tacs_solver.createNodeVec()
        tacs_solver.getNodes(xpts)
        x_s0 = xpts.getArray()

        x_a0 = vlm_builder.options['x_a0']

        self.add_subsystem('geometry_mapper',WingGeometry(xs=x_s0, xa=x_a0, y_knot=y_knot, LE_knot=LE_knot, TE_knot=TE_knot))

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
        
        for i in range(0,N_mp):
            mp.mphys_add_scenario('s'+str(i))

    def configure(self):

        # add parameters across the mp_groups: mp parameters, and AoA DV

        for i in range(0,N_mp):

            # add flow mp parameters

            for param in ['mach','q_inf','vel','mu']:
                self.mp_parameters.add_output(param+str(i), val = aero_parameters[param][i])
                self.connect(param+str(i), 'mp_group.s'+str(i)+'.aero.'+param)

            # add trim mp parameters
 
            param = 'load_factor'
            self.mp_parameters.add_output(param+str(i), val = trim_parameters[param][i])        
            self.connect(param+str(i),'mp_group.s'+str(i)+'.struct.inertial_loads.'+param)   
            self.connect(param+str(i),'mp_group.s'+str(i)+'.struct.fuel_loads.'+param)

            param = 'load_case_fuel_burned'
            self.mp_parameters.add_output(param+str(i), val = trim_parameters[param][i])
            self.connect(param+str(i),'mp_group.s'+str(i)+'.struct.fuel_loads.'+param)

            # add AoA DV
            
            param = 'alpha' 
            self.trim_dvs.add_output(param+str(i), val = aero_parameters[param][i])
            self.connect(param+str(i),'mp_group.s'+str(i)+'.aero.'+param)

        #self.mp_parameters.add_output('mach', val = aero_parameters['mach'])
        #self.connect('mach', 'mp_group.s0.aero.mach')
        #self.mp_parameters.add_output('q_inf', val = aero_parameters['q_inf'])
        #self.connect('q_inf', 'mp_group.s0.aero.q_inf')
        #self.mp_parameters.add_output('vel', val = aero_parameters['vel'])
        #self.connect('vel', 'mp_group.s0.aero.vel')
        #self.mp_parameters.add_output('mu', val = aero_parameters['mu'])
        #self.connect('mach', 'mp_group.s0.aero.mach')
        #self.connect('mu', 'mp_group.s0.aero.mu')

        # add trim mp parameters

        #self.mp_parameters.add_output('load_factor', val = trim_parameters['load_factor'])        
        #self.connect('load_factor','mp_group.s0.struct.inertial_loads.load_factor')   
        #self.connect('load_factor','mp_group.s0.struct.fuel_loads.load_factor')

        #self.mp_parameters.add_output('load_case_fuel_burned', val = trim_parameters['load_case_fuel_burned'])
        #self.connect('load_case_fuel_burned','mp_group.s0.struct.fuel_loads.load_case_fuel_burned')

        # add AoA DV

        #self.trim_dvs.add_output('alpha', val=0*np.pi/180.)
        #self.connect('alpha', 'mp_group.s0.aero.alpha')

        # add the structural thickness DVs

        initial_thickness = 0.01

        for comp, n in zip(['upper_skin','lower_skin','le_spar','te_spar','rib'],[patches.n_us,patches.n_ls,patches.n_le,patches.n_te,patches.n_rib]):
            if strutural_patch_lumping is False:                
                self.sizing_dvs.add_output(comp+'_thickness', val=initial_thickness, shape=n)
                self.connect(comp+'_thickness','struct_mapper.'+comp+'_thickness')
            else:
                self.sizing_dvs.add_output(comp+'_thickness_lumped',     val=initial_thickness, shape = 1)
                self.connect(comp+'_thickness_lumped','struct_lumping.'+comp+'_lumper.thickness_lumped')
                self.connect('struct_lumping.'+comp+'_lumper.thickness','struct_mapper.'+comp+'_thickness')

        # add the geometry DVs
        
        for comp, n in zip(['root_chord_delta','tip_chord_delta','tip_sweep_delta','span_delta','wing_thickness_delta','wing_twist_delta'],[1,1,1,1,len(y_knot),len(y_knot)-1]):
            self.geometric_dvs.add_output(comp, val=0.0, shape=n)
            self.connect(comp,'geometry_mapper.'+comp)

        ## add the fuel matching DV

        self.fuel_dvs.add_output('fuel_match', val=1.0)
        for i in range(0,N_mp):
            self.connect('fuel_match', 'mp_group.s'+str(i)+'.struct.fuel_loads.fuel_DV')        

        # connect the smoothness constraints

        for comp in ['upper_skin','lower_skin','le_spar','te_spar']:
            if strutural_patch_lumping is False:
                self.connect(comp+'_thickness','struct_smoothness.'+comp+'_smoothness.thickness')
            else:
                self.connect('struct_lumping.'+comp+'_lumper.thickness','struct_smoothness.'+comp+'_smoothness.thickness')

        # connect solver data

        for i in range(0,N_mp):
            self.connect('struct_mapper.dv_struct', 'mp_group.s'+str(i)+'.struct.dv_struct')

        # connect the geometry mesh outputs

        points = self.mp_group.mphys_add_coordinate_input()
        self.connect('geometry_mapper.x_s0_mesh','mp_group.struct_points')
        self.connect('geometry_mapper.x_a0_mesh','mp_group.aero_points')

   



# currently, summed loads are given to TACS, but inertial loads are set to zero load factor
# need to double-check that inertial loads look OK, compare with sand-alone, and that they look OK when there is no aero

# inertial and fuel loads are much slower on K than on your machine.  Maybe bring your stand-alon codes over to K, and test them out, with and without PBS
# same for the VLM.

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

#model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=False, rtol = 1E-9, atol=1E-10)
#model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True, aitken_max_factor=1.0, aitken_min_factor=0.3,rtol = 1E-7, atol=1E-10)
model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True ,rtol = 1E-7, atol=1E-10)
model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-9, atol=1e-10)

#model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True ,aitken_max_factor=0.8, rtol = 1E-7, atol=1E-10)
model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True , rtol = 1E-7, atol=1E-10)
model.mp_group.s1.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-9, atol=1e-10)

prob.run_model()

# use aitken is on now!  CS won't work with it on.
# and you relaxed the tolerance


#prob.check_totals(of=['mp_group.s0.aero.forces.CD', 'mp_group.s0.struct.mass.mass', 'mp_group.s0.struct.funcs.f_struct'], wrt=['alpha', 'span_delta'], method='cs')

#prob.check_totals(of=['mp_group.s0.aero.forces.CD', 'mp_group.s0.struct.mass.mass', 'mp_group.s0.struct.funcs.f_struct'], wrt=['alpha', 'upper_skin_thickness_lumped', 'lower_skin_thickness_lumped', 'le_spar_thickness_lumped', 'te_spar_thickness_lumped', 'rib_thickness_lumped', 'root_chord_delta', 'tip_chord_delta', 'tip_sweep_delta', 'span_delta', 'wing_thickness_delta', 'wing_twist_delta'], method='cs')

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
