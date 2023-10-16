import os

import numpy as np
import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from openaerostruct.geometry.utils import generate_vsp_surfaces
from openaerostruct.mphys import AeroBuilder
from funtofem.mphys import MeldBuilder
from tacs.mphys import TacsBuilder

import tacs_setup

class Top(Multipoint):
    def setup(self):
        # Input files
        vsp_file = os.path.join(os.path.dirname(__file__), "boeing777.vsp3")
        bdf_file = os.path.join(os.path.dirname(__file__), "boeing-777-wing.bdf")
        # Read the geometry.
        # VSP components to include in VLM mesh
        vsp_comps = ["Wing"]
        # Generate half-wing mesh of wing
        surfaces = generate_vsp_surfaces(vsp_file, symmetry=True, include=vsp_comps)

        # Default surface dictionary values
        default_dict = {
            # Wing definition
            "type": "aero",
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            # "twist_cp": np.zeros(3),  # Define twist using 3 B-spline cp's
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.0,  # CD of the surface at alpha=0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c": 0.12,  # thickness over chord ratio (NACA0015)
            "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
            # thickness
            "with_viscous": False,  # if true, compute viscous drag,
            "with_wave": False}

        #  Update our surface dict with defaults
        for surface in surfaces:
            surface.update(default_dict)

        mach = 0.85
        aoa = 1.0
        rho = 1.2
        yaw = 0.0
        vel = 178.
        re = 1e6

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('yaw', val=yaw, units='deg')
        dvs.add_output('rho', val=rho, units='kg/m**3')
        dvs.add_output('mach', mach)
        dvs.add_output('v', vel, units='m/s')
        dvs.add_output('reynolds', re, units="1/m")

        # OpenAeroStruct
        aero_builder = AeroBuilder(surfaces)
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero', aero_builder.get_mesh_coordinate_subsystem())

        # TACS
        tacs_options = {'element_callback' : tacs_setup.element_callback,
                        'problem_setup': tacs_setup.problem_setup,
                        'mesh_file': bdf_file}

        struct_builder = TacsBuilder(tacs_options, coupled=True)

        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem('mesh_struct', struct_builder.get_mesh_coordinate_subsystem())
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.02]))

        # MELD setup
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=1)
        ldxfer_builder.initialize(self.comm)

        # Scenario
        self.mphys_add_scenario('cruise', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                 struct_builder=struct_builder,
                                                                 ldxfer_builder=ldxfer_builder))

        for discipline in ['aero', 'struct']:
            self.mphys_connect_scenario_coordinate_source(
                'mesh_%s' % discipline, 'cruise', discipline)

        for dv in ['aoa', 'yaw', 'rho', 'mach', 'v', 'reynolds']:
            self.connect(dv, f'cruise.{dv}')
        self.connect('dv_struct', 'cruise.dv_struct')

prob = om.Problem()
prob.model = Top()

model = prob.model

model.add_design_var('dv_struct', lower=0.002, upper=0.2, scaler=1000.0)
model.add_design_var('aoa', lower=-10, upper=10.0, scaler=0.1)
model.add_objective('cruise.mass', scaler=1.0/1000.0)
model.add_constraint('cruise.ks_vmfailure', upper=1.0, scaler=1.0)
model.add_constraint('cruise.CL', equals=0.5, scaler=1.0)

prob.driver = om.ScipyOptimizeDriver(debug_print=['objs', 'nl_cons'], maxiter=200)
prob.driver.options['optimizer'] = 'SLSQP'

prob.setup()
om.n2(prob, show_browser=False, outfile='tacs_oas_aerostruct.html')

prob.run_driver()