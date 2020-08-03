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
from inertial_load_component import InertialLoads
from fuel_component import FuelMass, FuelLoads
from wing_area_component import WingArea, WingAreaComponent
from trim_component import Trim, FuelMatch
from flight_metric_components import FlightMetrics
from spar_depth_component import SparDepth


class Top(om.Group):

    def setup(self):

        # case setup inputs

        self.geometry_parameters = {
            'y_knot': np.array([0,3.0999,10.979510169492,16.968333898305,23.456226271186,29.44505]),                                    # y coordinates of knot locations
            'LE_knot': np.array([23.06635,25.251679291894,31.205055607314,35.736770145885000,40.641208661042000,45.16478]),             # x coordinates of LE knot locations
            'TE_knot': np.array([36.527275,37.057127604678,38.428622653948,41.498242368371000,44.820820782553000,47.887096577690000])   # x coordinates of TE knot locations
        }

        self.aero_parameters = {
            'mach': [0.85, .64],                            # mach number of each load case
            'q_inf': [12930., 28800.],#[9000., 12000.],#[12930., 28800.],                      # dynamic pressure of each load case, Pa
            'vel': [254., 217.6],                           # velocity of each load case, m/s
            'mu': [3.5E-5, 1.4E-5],                         # viscocity of each load case, 
            'alpha': np.array([1., 4.])*np.pi/180.,#np.array([0.5, 1.0])*np.pi/180.,#np.array([1., 4.])*np.pi/180.,         # AoA of each load case: this is a DV, so these values set the starting points
        }

        self.trim_parameters = {
            'load_factor': [1.0, 2.5],                      # load factor for each load case, L/W
            'load_case_fuel_burned': [.5, 1.0],             # fraction of FB expended at each load case
        }

        self.misc_parameters = {
            'structural_patch_lumping': False,              # reduces all the component thickness DVs into a single one: useful for checking totals
            'initial_thickness': 0.01,                      # starting thickness for each thickness DV, m
            'elastic_modulus': 73.1e9,                      # elastic modulus, Pa
            'poisson': 0.33,                                # poisson's ratio
            'k_corr': 5.0/6.0,                              # shear correction factor
            'ys': 324.0e6,                                  # yield stress, Pa
            'structural_density': 2780.0,                   # structural density, kg/m^3
            'gravity': 9.81,                                # gravitational constant, m/s^2
            'non_designable_weight': 14E5,                  # weight of everything but structure and fuel, N
            'cruise_range': 7725.*1852,                     # cruise range used to compute FB, m
            'reserve_fuel': 7500.,                          # reserve fuel not burned during cruise, kg
            'fuel_density':  810.,                          # fuel density, kg/m^3
            'TSFC': .53/3600,                               # TSFC used to compute FB
            'N_mp': 2,                                      # number of load cases
            'cruise_case_ID': 0,                            # load case ID which will be used to compute L/D
            'beta': .5,                                     # weighting between FB and LGW, to compute final objective function: beta = 1 is pure FB minimization
            'BDF_file': 'CRM_box_2nd.bdf',                  # BDF file used to define FEM 
            'VLM_mesh_file': 'CRM_VLM_mesh_extended.dat'    # file which contains the baseline VLM grid
        }

        # FEM patches, read from BDF

        self.patches = PatchList(self.misc_parameters['BDF_file'])
        self.patches.read_families()
        self.patches.create_DVs()

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
        aero_options['N_nodes'], aero_options['N_elements'], aero_options['x_a0'], aero_options['quad'] = read_VLM_mesh(self.misc_parameters['VLM_mesh_file'])

        # VLM builder

        vlm_builder = VLM_builder(aero_options)

        # TACS setup

        def add_elements(mesh):
            rho = self.misc_parameters['structural_density']         
            E = self.misc_parameters['elastic_modulus']            
            nu = self.misc_parameters['poisson']
            kcorr = self.misc_parameters['k_corr']
            ys = self.misc_parameters['ys']
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
                    'mesh_file'   : self.misc_parameters['BDF_file'],
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

        if self.misc_parameters['structural_patch_lumping'] is True:
            self.add_subsystem('struct_lumping',om.Group())
            for comp, n in zip(['upper_skin','lower_skin','le_spar','te_spar','rib'],[self.patches.n_us,self.patches.n_ls,self.patches.n_le,self.patches.n_te,self.patches.n_rib]):
                self.struct_lumping.add_subsystem(comp+'_lumper',LumpPatches(N=n)) 

        # structural mapper: map structural component arrays into the unified array that goes into tacs

        self.add_subsystem('struct_mapper',DesignPatches(patch_list=self.patches))

        # patch smoothness constraints

        self.add_subsystem('struct_smoothness',om.Group())
        for comp, n in zip(['upper_skin','lower_skin','le_spar','te_spar'],[self.patches.n_us,self.patches.n_ls,self.patches.n_le,self.patches.n_te]):        
            self.struct_smoothness.add_subsystem(comp+'_smoothness',PatchSmoothness(N=n))
        
        # geometry mapper

        struct_builder.init_solver(MPI.COMM_WORLD)
        tacs_solver = struct_builder.get_solver()
        xpts = tacs_solver.createNodeVec()
        tacs_solver.getNodes(xpts)
        x_s0 = xpts.getArray()

        x_a0 = vlm_builder.options['x_a0']

        self.add_subsystem('geometry_mapper',WingGeometry(
            xs=x_s0, 
            xa=x_a0, 
            y_knot=self.geometry_parameters['y_knot'], 
            LE_knot=self.geometry_parameters['LE_knot'], 
            TE_knot=self.geometry_parameters['TE_knot'])
        )

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
        
        for i in range(0,self.misc_parameters['N_mp']):
            mp.mphys_add_scenario('s'+str(i))

        # add inertial and fuel load components

        quad = np.zeros([tacs_solver.getNumElements(),4],'int')
        prop_ID = np.zeros(tacs_solver.getNumElements(),'int')
        for ielem, elem in enumerate(tacs_solver.getElements()):
            quad[ielem,:] = tacs_solver.getElementNodes(ielem)
            prop_ID[ielem] = elem.getComponentNum()

        self.mp_group.add_subsystem('non_aero_loads',om.Group())

        for i in range(0,self.misc_parameters['N_mp']):

            self.mp_group.non_aero_loads.add_subsystem('inertial_loads'+str(i),InertialLoads(
                N_nodes=int(len(x_s0)/3), 
                elements=quad, 
                prop_ID=prop_ID, 
                n_dvs=len(self.patches.families), 
                rho=self.misc_parameters['structural_density'], 
                gravity=self.misc_parameters['gravity'])
            )

            self.mp_group.non_aero_loads.add_subsystem('fuel_loads'+str(i),FuelLoads(
                N_nodes=int(len(x_s0)/3), 
                elements=quad, 
                prop_ID=prop_ID, 
                patches=self.patches, 
                gravity=self.misc_parameters['gravity'], 
                reserve_fuel=self.misc_parameters['reserve_fuel'], 
                fuel_density=self.misc_parameters['fuel_density'])
            )

        # create a group to hold the various output parameters that don't belong anywhere else

        self.add_subsystem('outputs',om.Group())
        
        # add a component to compute wing area

        self.outputs.add_subsystem('wing_area',WingAreaComponent(
            N_nodes=aero_options['N_nodes'], 
            quad=aero_options['quad'])
        )

        # add trim components

        for i in range(0,self.misc_parameters['N_mp']):
            self.outputs.add_subsystem('trim'+str(i),Trim(
                non_designable_weight=self.misc_parameters['non_designable_weight'],
                gravity=self.misc_parameters['gravity'])
            )

        # add a component which computes flight metrics: FB, TOGW, LGW

        self.outputs.add_subsystem('flight_metrics',FlightMetrics(
            non_designable_weight=self.misc_parameters['non_designable_weight'], 
            range=self.misc_parameters['cruise_range'], 
            TSFC=self.misc_parameters['TSFC'], 
            beta=self.misc_parameters['beta'],
            gravity=self.misc_parameters['gravity'])
        )

        # add a component which computes the total available fuel mass: available_fuel_mass*fuel_DV is the fuel mass actually being used

        self.outputs.add_subsystem('available_fuel_mass', om.ExecComp('available_fuel_mass=fuel_mass/fuel_DV'))

        # add a component which computes the mis-match in the FB output and fuel_DV input

        self.outputs.add_subsystem('fuel_match',FuelMatch())

        # add a component which computes the minimum TE spar depth

        self.outputs.add_subsystem('spar_depth',SparDepth(
            N_nodes=int(len(x_s0)/3), 
            elements=quad, 
            prop_ID=prop_ID, 
            patches=self.patches)
        )

    def configure(self):

        # add parameters across the mp_groups: mp parameters, and AoA DV

        for i in range(0,self.misc_parameters['N_mp']):

            # add flow mp parameters

            for param in ['mach','q_inf','vel','mu']:
                self.mp_parameters.add_output(param+str(i), val = self.aero_parameters[param][i])
                self.connect(param+str(i), 'mp_group.s'+str(i)+'.aero.'+param)

            # add trim mp parameters
 
            param = 'load_factor'
            self.mp_parameters.add_output(param+str(i), val = self.trim_parameters[param][i])        
            self.connect(param+str(i),'mp_group.non_aero_loads.inertial_loads'+str(i)+'.'+param)
            self.connect(param+str(i),'mp_group.non_aero_loads.fuel_loads'+str(i)+'.'+param)

            param = 'load_case_fuel_burned'
            self.mp_parameters.add_output(param+str(i), val = self.trim_parameters[param][i])
            self.connect(param+str(i),'mp_group.non_aero_loads.fuel_loads'+str(i)+'.'+param)

            # add AoA DV
            
            param = 'alpha' 
            self.trim_dvs.add_output(param+str(i), val = self.aero_parameters[param][i])
            self.connect(param+str(i),'mp_group.s'+str(i)+'.aero.'+param)

        # add the structural thickness DVs

        for comp, n in zip(['upper_skin','lower_skin','le_spar','te_spar','rib'],[self.patches.n_us,self.patches.n_ls,self.patches.n_le,self.patches.n_te,self.patches.n_rib]):
            if self.misc_parameters['structural_patch_lumping'] is False:                
                self.sizing_dvs.add_output(comp+'_thickness', val=self.misc_parameters['initial_thickness'], shape=n)
                self.connect(comp+'_thickness','struct_mapper.'+comp+'_thickness')
            else:
                self.sizing_dvs.add_output(comp+'_thickness_lumped',     val=self.misc_parameters['initial_thickness'], shape = 1)
                self.connect(comp+'_thickness_lumped','struct_lumping.'+comp+'_lumper.thickness_lumped')
                self.connect('struct_lumping.'+comp+'_lumper.thickness','struct_mapper.'+comp+'_thickness')

        # add the geometry DVs
        
        for comp, n in zip(['root_chord_delta','tip_chord_delta','tip_sweep_delta','span_delta','wing_thickness_delta','wing_twist_delta'],[1,1,1,1,len(self.geometry_parameters['y_knot']),len(self.geometry_parameters['y_knot'])-1]):
            self.geometric_dvs.add_output(comp, val=0.0, shape=n)
            self.connect(comp,'geometry_mapper.'+comp)

        ## add the fuel matching DV

        self.fuel_dvs.add_output('fuel_dv', val=1.0)
        for i in range(0,self.misc_parameters['N_mp']):
            self.connect('fuel_dv', 'mp_group.non_aero_loads.fuel_loads'+str(i)+'.fuel_DV')

        # connect the smoothness constraints

        for comp in ['upper_skin','lower_skin','le_spar','te_spar']:
            if self.misc_parameters['structural_patch_lumping'] is False:
                self.connect(comp+'_thickness','struct_smoothness.'+comp+'_smoothness.thickness')
            else:
                self.connect('struct_lumping.'+comp+'_lumper.thickness','struct_smoothness.'+comp+'_smoothness.thickness')

        # connect solver data

        for i in range(0,self.misc_parameters['N_mp']):
            self.connect('struct_mapper.dv_struct', 'mp_group.s'+str(i)+'.struct.dv_struct')

        # connect the inertial/fuel load geometry and dv_struct inputs, and connect the outputs to the load summer

        for i in range(0,self.misc_parameters['N_mp']):
            self.connect('geometry_mapper.x_s0_mesh','mp_group.non_aero_loads.inertial_loads'+str(i)+'.x_s0')
            self.connect('struct_mapper.dv_struct','mp_group.non_aero_loads.inertial_loads'+str(i)+'.dv_struct')
            self.connect('geometry_mapper.x_s0_mesh','mp_group.non_aero_loads.fuel_loads'+str(i)+'.x_s0')

            self.connect('mp_group.non_aero_loads.inertial_loads'+str(i)+'.F_inertial','mp_group.s'+str(i)+'.struct.sum_loads.F_inertial')
            self.connect('mp_group.non_aero_loads.fuel_loads'+str(i)+'.F_fuel','mp_group.s'+str(i)+'.struct.sum_loads.F_fuel')

        # connect the geometry mesh outputs

        points = self.mp_group.mphys_add_coordinate_input()
        self.connect('geometry_mapper.x_s0_mesh','mp_group.struct_points')
        self.connect('geometry_mapper.x_a0_mesh','mp_group.aero_points')

        # connect the wing area module

        self.connect('mp_group.aero_mesh.x_a0','outputs.wing_area.x') 

        # connect the trim components

        for i in range(0,self.misc_parameters['N_mp']):
            self.connect('outputs.wing_area.area','outputs.trim'+str(i)+'.wing_area')
            self.connect('mp_group.s'+str(i)+'.aero.CL','outputs.trim'+str(i)+'.CL')
            self.connect('mp_group.s'+str(i)+'.struct.mass','outputs.trim'+str(i)+'.structural_mass')
            self.connect('mp_group.non_aero_loads.fuel_loads'+str(i)+'.fuel_mass','outputs.trim'+str(i)+'.fuel_mass')
            self.connect('q_inf'+str(i),'outputs.trim'+str(i)+'.q_inf')

        # connect the flight metric components
        
        self.connect('mp_group.s'+str(self.misc_parameters['cruise_case_ID'])+'.aero.CL','outputs.flight_metrics.CL')
        self.connect('mp_group.s'+str(self.misc_parameters['cruise_case_ID'])+'.aero.CD','outputs.flight_metrics.CD')
        i = self.trim_parameters['load_case_fuel_burned'].index(1.)  # want to use a full fuel weight for TOGW computation
        self.connect('mp_group.non_aero_loads.fuel_loads'+str(i)+'.fuel_mass','outputs.flight_metrics.fuel_mass')
        self.connect('mp_group.s'+str(i)+'.struct.mass','outputs.flight_metrics.structural_mass')
        self.connect('vel'+str(self.misc_parameters['cruise_case_ID']),'outputs.flight_metrics.velocity')

        # connect the component which computes the available fuel mass

        i = self.trim_parameters['load_case_fuel_burned'].index(1.)  # want to use a full fuel weight for this computation
        self.connect('mp_group.non_aero_loads.fuel_loads'+str(i)+'.fuel_mass','outputs.available_fuel_mass.fuel_mass')
        self.connect('fuel_dv','outputs.available_fuel_mass.fuel_DV')

        # connect the fuel match components

        self.connect('outputs.available_fuel_mass.available_fuel_mass','outputs.fuel_match.fuel_mass')
        self.connect('outputs.flight_metrics.FB','outputs.fuel_match.fuel_burn')
        self.connect('fuel_dv','outputs.fuel_match.fuel_DV')

        # connect the spar depth components

        self.connect('mp_group.struct_mesh.x_s0','outputs.spar_depth.x')        




################################################################################
# OpenMDAO setup
################################################################################

prob = om.Problem()
prob.model = Top()
model = prob.model

model.nonlinear_solver = om.NonlinearRunOnce()
model.linear_solver = om.LinearRunOnce()




## Use this if you want to check totals: need to use CS versions of TACS and MELD.  Also, can't use_aitken for this to work.  And since you can't use_aitken, q_inf has to be relatively low

#prob.setup(mode='rev',force_alloc_complex=True)
#om.n2(prob, show_browser=False, outfile='CRM_mphys_as_vlm.html')

#model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=False, rtol = 1E-11, atol=1E-11)
#model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-11, atol=1e-11)

#model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=False, rtol = 1E-11, atol=1E-11)
#model.mp_group.s1.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-11, atol=1e-11)

#prob.run_model()

##prob.check_totals(of=['mp_group.s0.aero.forces.CD'], wrt=['alpha0'], method='cs')
##print(prob['mp_group.s0.aero.forces.CD'])
##print(prob['alpha0'])
##prob.check_totals(of=['CD'], wrt=['alpha0'], method='cs')

#prob.check_totals(of=['mp_group.s1.struct.funcs.f_struct'], wrt=['alpha1'], method='cs')

## Use this if you don't want to check totals

prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='CRM_mphys_as_vlm.html')

model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True ,rtol = 1E-7, atol=1E-10)
model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-7, atol=1e-10)

model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True , rtol = 1E-7, atol=1E-10)
model.mp_group.s1.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-7, atol=1e-10)

prob.run_model()




# move that q back!!!!!  and aoa



#prob.check_totals(of=['mp_group.s0.aero.forces.CD', 'mp_group.s0.struct.mass.mass', 'mp_group.s0.struct.funcs.f_struct'], wrt=['alpha', 'span_delta'], method='cs')

#prob.check_totals(of=['mp_group.s0.aero.forces.CD', 'mp_group.s0.struct.mass.mass', 'mp_group.s0.struct.funcs.f_struct'], wrt=['alpha', 'upper_skin_thickness_lumped', 'lower_skin_thickness_lumped', 'le_spar_thickness_lumped', 'te_spar_thickness_lumped', 'rib_thickness_lumped', 'root_chord_delta', 'tip_chord_delta', 'tip_sweep_delta', 'span_delta', 'wing_thickness_delta', 'wing_twist_delta'], method='cs')




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
