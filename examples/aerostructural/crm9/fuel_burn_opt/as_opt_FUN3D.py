#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from tacs import elements, constitutive, functions, TACS

from mphys import Multipoint
from sfe.mphys import Fun3dSfeBuilder
from mphys.solver_builders.mphys_tacs import TACS_builder
from mphys.solver_builders.mphys_modal_solver import ModalBuilder
from mphys.solver_builders.mphys_meld import MeldBuilder

from structural_patches_component import PatchList, DesignPatches, PatchSmoothness, LumpPatches
from wing_geometry_component import WingGeometry, airfoil_thickness_bounds
from inertial_load_component import InertialLoads
from fuel_component import FuelMass, FuelLoads
from wing_area_component import WingArea, WingAreaComponent
from trim_component import Trim, FuelMatch
from flight_metric_components import FlightMetrics
from spar_depth_component import SparDepth

comm = MPI.COMM_WORLD
rank = comm.rank

class Top(om.Group):

    def setup(self):

        # case setup inputs

        self.geometry_parameters = {
            'y_knot': np.array([0,3.0999,10.979510169492,16.968333898305,23.456226271186,29.44505]),                                    # y coordinates of knot locations
            'LE_knot': np.array([23.06635,25.251679291894,31.205055607314,35.736770145885000,40.641208661042000,45.16478]),             # x coordinates of LE knot locations
            'TE_knot': np.array([36.527275,37.057127604678,38.428622653948,41.498242368371000,44.820820782553000,47.887096577690000])   # x coordinates of TE knot locations
        }

        self.aero_parameters = {
            'mach': [0.85, .64],                                   # mach number of each load case
            'q_inf': [12930., 28800.],                             # dynamic pressure of each load case, Pa
            'vel': [254., 217.6],                                  # velocity of each load case, m/s
            'mu': [3.5E-5, 1.4E-5],                                # viscocity of each load case,
            'reynolds': [254./3.5E-5, 217.6/1.4E-5],        # Re-per-length of each load case
            'aoa': [1., 4.],                                       # AoA of each load case: this is a DV, so these values set the starting points
        }

        self.trim_parameters = {
            'load_factor': [1.0, 2.5],                             # load factor for each load case, L/W
            'load_case_fuel_burned': [.5, 1.0],                    # fraction of FB expended at each load case
        }

        self.misc_parameters = {
            'structural_patch_lumping': True,#False,                     # reduces all the component thickness DVs into a single one: useful for checking totals
            'initial_thickness': 0.02,                             # starting thickness for each thickness DV, m
            'elastic_modulus': 73.1e9,                             # elastic modulus, Pa
            'poisson': 0.33,                                       # poisson's ratio
            'k_corr': 5.0/6.0,                                     # shear correction factor
            'ys': 324.0e6,                                         # yield stress, Pa
            'structural_density': 2780.0,                          # structural density, kg/m^3
            'gravity': 9.81,                                       # gravitational constant, m/s^2
            'non_designable_weight': 14E5,                         # weight of everything but structure and fuel, N
            'cruise_range': 7725.*1852,                            # cruise range used to compute FB, m
            'reserve_fuel': 7500.,                                 # reserve fuel not burned during cruise, kg
            'fuel_density':  810.,                                 # fuel density, kg/m^3
            'TSFC': .53/3600,                                      # TSFC used to compute FB
            'N_mp': 2,                                             # number of load cases
            'cruise_case_ID': 0,                                   # load case ID which will be used to compute L/D
            'beta': 1.,                                            # weighting between FB and LGW, to compute final objective function: beta = 1 is pure FB minimization, 0 is pure LGW
            'BDF_file': 'CRM_box_2nd.bdf',                         # BDF file used to define FEM
            'FUN3D_mesh_file': 'crm_rans_coarse.b8.ugrid',           # file which contains the baseline FUN3D grid
            'FUN3D_boundary_list': [3],                            # wing boundary ID in mapbc
        }

        self.opt_parameters = {
            'min_thickness': 0.003,                                # minimum thickness for each thickness DV, m
            'max_thickness': 0.03,                                 # maximum thickness for each thickness DV, m
            'delta_thickness': 0.001,                              # bound on smoothness constraints, m
            'root_chord_bounds': [-.01, .01],#[-3.0, 3.0],                      # plus-minus bounds on root-chord delta, m
            'tip_chord_bounds': [-.01, .01],#[-1.0, 1.0],                       # plus-minus bounds on tip-chord delta, m
            'tip_sweep_bounds': [-.01, .01],#[-10., 10.],                       # plus-minus bounds on tip-sweep delta, m
            'span_extend_bounds': [-.01, .01],#[-10., 10.],                     # plus-minus bounds on span-extension delta, m
            'allowable_airfoil_thickness_fraction': 0.5,           # fraction of baseline airfoil thickness allowable for plus-minus bounds, at each span station
            'wing_twist_bounds': np.array([-10., 10.])*np.pi/180., # plus-minus bounds on wing twist at each span station, rad
            'func_struct_bound': 2.0/3.0,                          # upper allowable bound on func_struct
            'spar_depth_bound': 0.19,                              # lower allowable bound on spar depth, m
        }

        # FEM patches, read from BDF

        self.patches = PatchList(self.misc_parameters['BDF_file'])
        self.patches.read_families()
        self.patches.create_DVs()

        # FUN3D builder

        aero_builder = Fun3dSfeBuilder(self.misc_parameters['FUN3D_mesh_file'],self.misc_parameters['FUN3D_boundary_list'])

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

        struct_builder = TACS_builder(tacs_setup, check_partials=False)

        meld_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}

        # MELD builder

        meld_builder = MeldBuilder(meld_options, aero_builder, struct_builder, check_partials=False)

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
            self.struct_smoothness.add_subsystem(comp+'_smoothness',PatchSmoothness(N=n, delta=self.opt_parameters['delta_thickness']))

        # geometry mapper

        aero_builder.init_solver(MPI.COMM_WORLD)
        x,y,z = aero_builder.meshdef.get_global_image_of_coordinates_on_boundary(self.misc_parameters['FUN3D_boundary_list'],bcast=True)
        x_a0 = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1)),z.reshape((-1,1))),axis=1).flatten(order="C")
        connect = aero_builder.meshdef.get_global_image_of_boundary_connectivity(self.misc_parameters['FUN3D_boundary_list'],bcast=True)
        aero_surface_connect = np.zeros([len(connect)*2,3],dtype=int)

        count = 0
        for i in range(0,len(connect)):
            if len(connect[i]) > 3:
                aero_surface_connect[count,:] = connect[i][[0,1,2]] + 1
                aero_surface_connect[count+1,:] = connect[i][[0,2,3]] + 1
                count = count + 2
            else:
                aero_surface_connect[count,:] = connect[i] + 1
                count = count + 1
        aero_surface_connect = aero_surface_connect[0:count,:]

        struct_builder.init_solver(MPI.COMM_WORLD)
        tacs_solver = struct_builder.get_solver()
        xpts = tacs_solver.createNodeVec()
        tacs_solver.getNodes(xpts)
        x_s0 = xpts.getArray()

        x_s0 = aero_builder.meshdef._gatherv_vector(x_s0)
        comm = MPI.COMM_WORLD
        x_s0 = comm.bcast(x_s0)

        self.add_subsystem('geometry_mapper',WingGeometry(
            xs=x_s0,
            xa=x_a0,
            y_knot=self.geometry_parameters['y_knot'],
            LE_knot=self.geometry_parameters['LE_knot'],
            TE_knot=self.geometry_parameters['TE_knot'])
        )

        self.min_airfoil_thickness, self.max_airfoil_thickness = airfoil_thickness_bounds(
            xs=x_s0,
            y_knot=self.geometry_parameters['y_knot'],
            airfoil_thickness_fraction=self.opt_parameters['allowable_airfoil_thickness_fraction']
        )

        # add a component to compute wing area: VLM had this in outputs, but have to move it up to model level here, b/c aero_funcs needs it

        self.add_subsystem('wing_area',WingAreaComponent(
            N_nodes=int(len(x_a0)/3),
            connect=aero_surface_connect)
        )

        self.baseline_wing_area = WingArea(
            nodes=x_a0,
            connect=aero_surface_connect
        )
        self.baseline_wing_area.compute()

        # each AS_Multipoint instance can keep multiple points with the same formulation

        mp = self.add_subsystem(
            'mp_group',
            Multipoint(
                aero_builder   = aero_builder,
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

        quad = aero_builder.meshdef._gatherv_vector(quad.flatten())
        quad = comm.bcast(quad)
        quad = np.reshape(quad,(int(len(quad)/4),4))
        prop_ID = aero_builder.meshdef._gatherv_vector(prop_ID)
        prop_ID = comm.bcast(prop_ID)

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

        self.baseline_available_fuel = FuelMass(
            nodes=x_s0,
            quads=quad,
            prop_ID=prop_ID,
            patches=self.patches,
            fuel_density=self.misc_parameters['fuel_density']
        )
        self.baseline_available_fuel.compute()

        # create a group to hold the various output parameters that don't belong anywhere else

        self.add_subsystem('outputs',om.Group())

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

            for param in ['mach','vel','q_inf','reynolds']:
                self.mp_parameters.add_output(param+str(i), val = self.aero_parameters[param][i])

            for param in ['mach','reynolds']:
                self.connect(param+str(i), 'mp_group.s'+str(i)+'.aero.flow.'+param)

            for param in ['mach','reynolds','q_inf']:
                self.connect(param+str(i), 'mp_group.s'+str(i)+'.aero.forces.'+param)

            for param in ['q_inf']:
                self.connect(param+str(i), 'mp_group.s'+str(i)+'.aero_funcs.'+param)

            # add trim mp parameters

            param = 'load_factor'
            self.mp_parameters.add_output(param+str(i), val = self.trim_parameters[param][i])
            self.connect(param+str(i),'mp_group.non_aero_loads.inertial_loads'+str(i)+'.'+param)
            self.connect(param+str(i),'mp_group.non_aero_loads.fuel_loads'+str(i)+'.'+param)

            param = 'load_case_fuel_burned'
            self.mp_parameters.add_output(param+str(i), val = self.trim_parameters[param][i])
            self.connect(param+str(i),'mp_group.non_aero_loads.fuel_loads'+str(i)+'.'+param)

            # add AoA DV

            param = 'aoa'
            self.trim_dvs.add_output(param+str(i), val = self.aero_parameters[param][i], units='deg')
            self.connect(param+str(i),'mp_group.s'+str(i)+'.aero.flow.'+param)
            self.connect(param+str(i),'mp_group.s'+str(i)+'.aero_funcs.'+param)

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
            self.connect('geometry_mapper.x_struct0_mesh','mp_group.non_aero_loads.inertial_loads'+str(i)+'.x_struct0')
            self.connect('struct_mapper.dv_struct','mp_group.non_aero_loads.inertial_loads'+str(i)+'.dv_struct')
            self.connect('geometry_mapper.x_struct0_mesh','mp_group.non_aero_loads.fuel_loads'+str(i)+'.x_struct0')

            self.connect('mp_group.non_aero_loads.inertial_loads'+str(i)+'.F_inertial','mp_group.s'+str(i)+'.struct.sum_loads.F_inertial')
            self.connect('mp_group.non_aero_loads.fuel_loads'+str(i)+'.F_fuel','mp_group.s'+str(i)+'.struct.sum_loads.F_fuel')

        # connect the geometry mesh outputs

        points = self.mp_group.mphys_add_coordinate_input()
        self.connect('geometry_mapper.x_struct0_mesh','mp_group.struct_points')
        self.connect('geometry_mapper.x_aero0_mesh','mp_group.aero_points')

        # connect the wing area module

        self.connect('geometry_mapper.x_aero0_mesh','wing_area.x')

        for i in range(0,self.misc_parameters['N_mp']):
            self.connect('wing_area.area','mp_group.s'+str(i)+'.aero_funcs.ref_area')

        # connect the trim components

        for i in range(0,self.misc_parameters['N_mp']):
            self.connect('wing_area.area','outputs.trim'+str(i)+'.wing_area')
            self.connect('mp_group.s'+str(i)+'.aero_funcs.C_L','outputs.trim'+str(i)+'.CL')
            self.connect('mp_group.s'+str(i)+'.struct.mass','outputs.trim'+str(i)+'.structural_mass')
            self.connect('mp_group.non_aero_loads.fuel_loads'+str(i)+'.fuel_mass','outputs.trim'+str(i)+'.fuel_mass')
            self.connect('q_inf'+str(i),'outputs.trim'+str(i)+'.q_inf')

        # connect the flight metric components

        self.connect('mp_group.s'+str(self.misc_parameters['cruise_case_ID'])+'.aero_funcs.C_L','outputs.flight_metrics.CL')
        self.connect('mp_group.s'+str(self.misc_parameters['cruise_case_ID'])+'.aero_funcs.C_D','outputs.flight_metrics.CD')
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

        self.connect('mp_group.struct_mesh.x_struct0','outputs.spar_depth.x')




################################################################################
# OpenMDAO setup
################################################################################

prob = om.Problem()
prob.model = Top()
model = prob.model

model.nonlinear_solver = om.NonlinearRunOnce()
model.linear_solver = om.LinearRunOnce()




# Use this if you want to check totals: CS of FUN3D/SFE not availble for now, so can use aitken to do this
# Also, turn patch_lumping on.

prob.setup(mode='rev')

model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=12, iprint=2, use_aitken=True ,rtol = 1E-7, atol=1E-10)
model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=12, iprint=2, rtol = 1e-7, atol=1e-10)

model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=12, iprint=2, use_aitken=True , rtol = 1E-7, atol=1E-10)
model.mp_group.s1.linear_solver = om.LinearBlockGS(maxiter=12, iprint=2, rtol = 1e-7, atol=1e-10)

prob.run_model()

boo

prob.check_totals(
        of=[
        #'mp_group.s1.struct.funcs.func_struct',
        #'wing_area.area',
        #'outputs.trim0.load_factor',
        'outputs.trim1.load_factor',
        #'outputs.flight_metrics.FB',
        #'outputs.flight_metrics.LGW',
        #'outputs.flight_metrics.final_objective',
        #'outputs.available_fuel_mass.available_fuel_mass',
        #'outputs.fuel_match.fuel_mismatch',
        #'outputs.spar_depth.spar_depth',
        ],
        wrt=[
        #'aoa0',
        #'aoa1',
        'upper_skin_thickness_lumped',
        #'lower_skin_thickness_lumped',
        #'le_spar_thickness_lumped',
        #'te_spar_thickness_lumped',
        #'rib_thickness_lumped',
        #'root_chord_delta',
        #'tip_chord_delta',
        #'tip_sweep_delta',
        #'span_delta',
        #'wing_thickness_delta',
        #'wing_twist_delta',
        #'fuel_dv'
        ], method='fd', step=1e-06)


boo


## Use this if you don't want to check totals, but just want to run the model once

prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='CRM_mphys_as_fun3d.html')

model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True ,rtol = 1E-7, atol=1E-10)
model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-7, atol=1e-10)

model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True , rtol = 1E-7, atol=1E-10)
model.mp_group.s1.linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, rtol = 1e-7, atol=1e-10)

prob.run_model()

## add design variables

prob.setup(mode='rev')
om.n2(prob, show_browser=False, outfile='CRM_mphys_as_fun3d.html')

prob.model.add_design_var('upper_skin_thickness',  lower=model.opt_parameters['min_thickness'],         upper=model.opt_parameters['max_thickness'],         ref=model.misc_parameters['initial_thickness'])
prob.model.add_design_var('lower_skin_thickness',  lower=model.opt_parameters['min_thickness'],         upper=model.opt_parameters['max_thickness'],         ref=model.misc_parameters['initial_thickness'])
prob.model.add_design_var('le_spar_thickness',     lower=model.opt_parameters['min_thickness'],         upper=model.opt_parameters['max_thickness'],         ref=model.misc_parameters['initial_thickness'])
prob.model.add_design_var('te_spar_thickness',     lower=model.opt_parameters['min_thickness'],         upper=model.opt_parameters['max_thickness'],         ref=model.misc_parameters['initial_thickness'])
prob.model.add_design_var('rib_thickness',         lower=model.opt_parameters['min_thickness'],         upper=model.opt_parameters['max_thickness'],         ref=model.misc_parameters['initial_thickness'])
prob.model.add_design_var('root_chord_delta',      lower=model.opt_parameters['root_chord_bounds'][0],  upper=model.opt_parameters['root_chord_bounds'][1],  ref=1.0)
prob.model.add_design_var('tip_chord_delta',       lower=model.opt_parameters['tip_chord_bounds'][0],   upper=model.opt_parameters['tip_chord_bounds'][1],   ref=1.0)
prob.model.add_design_var('tip_sweep_delta',       lower=model.opt_parameters['tip_sweep_bounds'][0],   upper=model.opt_parameters['tip_sweep_bounds'][1],   ref=1.0)
prob.model.add_design_var('span_delta',            lower=model.opt_parameters['span_extend_bounds'][0], upper=model.opt_parameters['span_extend_bounds'][1], ref=1.0)
prob.model.add_design_var('wing_thickness_delta',  lower=model.min_airfoil_thickness,                   upper=model.max_airfoil_thickness,                   ref=1.0)
prob.model.add_design_var('wing_twist_delta',      lower=model.opt_parameters['wing_twist_bounds'][0],  upper=model.opt_parameters['wing_twist_bounds'][1],  ref=1.0*np.pi/180)
prob.model.add_design_var('fuel_dv',               lower=0.0,                                           upper=1.0,                                           ref=1.0)

for i in range(0,model.misc_parameters['N_mp']):
    prob.model.add_design_var('aoa'+str(i),         lower=-10,                                           upper=10          ,                                 ref=1.0, units='deg')

## add sizing smoothness constraints

prob.model.add_constraint('struct_smoothness.upper_skin_smoothness.diff', ref=model.opt_parameters['delta_thickness'], upper=0.0, linear=True)
prob.model.add_constraint('struct_smoothness.lower_skin_smoothness.diff', ref=model.opt_parameters['delta_thickness'], upper=0.0, linear=True)
prob.model.add_constraint('struct_smoothness.le_spar_smoothness.diff',    ref=model.opt_parameters['delta_thickness'], upper=0.0, linear=True)
prob.model.add_constraint('struct_smoothness.te_spar_smoothness.diff',    ref=model.opt_parameters['delta_thickness'], upper=0.0, linear=True)

## add func_struct constraints, for every scenario except cruise

for i in range(0,model.misc_parameters['N_mp']):
    if i != model.misc_parameters['cruise_case_ID']:
        prob.model.add_constraint('mp_group.s'+str(i)+'.struct.funcs.func_struct', ref=1.0, upper=model.opt_parameters['func_struct_bound'])

## add trim constraints

for i in range(0,model.misc_parameters['N_mp']):
    prob.model.add_constraint('outputs.trim'+str(i)+'.load_factor', ref=1.0, equals=model.trim_parameters['load_factor'][i])

## add fuel mismatch constraint

prob.model.add_constraint('outputs.fuel_match.fuel_mismatch', ref=1.0, equals=0.0)

## add wing area constraint

prob.model.add_constraint('outputs.wing_area.area', ref=model.baseline_wing_area.A, lower=model.baseline_wing_area.A)

## add available fuel mass constraint

prob.model.add_constraint('outputs.available_fuel_mass.available_fuel_mass', ref=np.sum(model.baseline_available_fuel.mass), lower=np.sum(model.baseline_available_fuel.mass))

## add spar depth constraint

prob.model.add_constraint('outputs.spar_depth.spar_depth', ref=model.opt_parameters['spar_depth_bound'], lower=model.opt_parameters['spar_depth_bound'])

## add objective function

prob.model.add_objective('outputs.flight_metrics.final_objective',ref=1.0)

## set driver options, and then run driver

prob.driver = om.ScipyOptimizeDriver(debug_print=['ln_cons','nl_cons','objs','totals'])
#prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-4
prob.driver.options['disp'] = True
prob.driver.options['maxiter'] = 200

prob.driver.recording_options['includes'] = ['*']
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True

recorder = om.SqliteRecorder("cases_VLM.sql")
prob.driver.add_recorder(recorder)

prob.setup(mode='rev')

model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True ,rtol = 1E-7, atol=1E-7)
model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=200, iprint=2, rtol = 1e-6, atol=1e-6)

model.mp_group.s1.nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=True , rtol = 1E-7, atol=1E-7)
model.mp_group.s1.linear_solver = om.LinearBlockGS(maxiter=200, iprint=2, rtol = 1e-6, atol=1e-6)

prob.run_driver()

## write out data

cr = om.CaseReader("cases_FUN3D.sql")
driver_cases = cr.list_cases('driver')

case = cr.get_case(0)
cons = case.get_constraints()
dvs = case.get_design_vars()
objs = case.get_objectives()

f = open("history_FUN3D.dat","w+")

for i, k in enumerate(objs.keys()):
    f.write('objective: ' + k + '\n')
    for j, case_id in enumerate(driver_cases):
        f.write(str(j) + ' ' + str(cr.get_case(case_id).get_objectives(scaled=False)[k][0]) + '\n')
    f.write(' ' + '\n')

for i, k in enumerate(cons.keys()):
    f.write('constraint: ' + k + '\n')
    for j, case_id in enumerate(driver_cases):
        f.write(str(j) + ' ' + ' '.join(map(str,cr.get_case(case_id).get_constraints(scaled=False)[k])) + '\n')
    f.write(' ' + '\n')

for i, k in enumerate(dvs.keys()):
    f.write('DV: ' + k + '\n')
    for j, case_id in enumerate(driver_cases):
        f.write(str(j) + ' ' + ' '.join(map(str,cr.get_case(case_id).get_design_vars(scaled=False)[k])) + '\n')
    f.write(' ' + '\n')

f.close()
