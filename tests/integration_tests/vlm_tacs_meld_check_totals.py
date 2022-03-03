from __future__ import print_function, division
from mpi4py import MPI
import numpy as np

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from vlm_solver.mphys_vlm import VlmBuilder
from tacs.mphys import TacsBuilder
from mphys.solver_builders.mphys_meld import MeldBuilder

from tacs import elements, constitutive, functions

use_modal = False

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    rho = 2780.0  # density, kg/m^3
    E = 73.1e9  # elastic modulus, Pa
    nu = 0.33  # poisson's ratio
    ys = 324.0e6  # yield stress, Pa
    thickness = 0.003
    min_thickness = 0.002
    max_thickness = 0.05

    # Setup (isotropic) property and constitutive objects
    prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
    # Set one thickness dv for every component
    con = constitutive.IsoShellConstitutive(prop, t=thickness, tNum=dvNum, tlb=min_thickness, tub=max_thickness)

    # For each element type in this component,
    # pass back the appropriate tacs element object
    transform = None
    elem = elements.Quad4Shell(transform, con)

    return elem

def problem_setup(scenario_name, fea_assembler, problem):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """
    # Add TACS Functions
    # Only include mass from elements that belong to pytacs components (i.e. skip concentrated masses)
    problem.addFunction('mass', functions.StructuralMass)
    problem.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

    # Add gravity load
    g = np.array([0.0, 0.0, -9.81])  # m/s^2
    problem.addInertialLoad(g)

class Top(Multipoint):
    def setup(self):
        # VLM
        mesh_file = '../input_files/debug_VLM.dat'
        mach = 0.85,
        aoa = 1.0
        q_inf = 25000.
        vel = 178.
        nu = 3.5E-5

        aero_builder = VlmBuilder(mesh_file, complex_step=True)
        aero_builder.initialize(self.comm)

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('nu', nu)

        self.add_subsystem('mesh_aero', aero_builder.get_mesh_coordinate_subsystem())

        # TACS
        tacs_options = {'element_callback' : element_callback,
                        "problem_setup": problem_setup,
                        'mesh_file': '../input_files/debug.bdf'}

        if use_modal:
            tacs_options['nmodes'] = 15
            #struct_assembler = ModalStructAssembler(tacs_options)
        else:
            struct_builder = TacsBuilder(tacs_options, check_partials=True, coupled=True, write_solution=False)

        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem('mesh_struct', struct_builder.get_mesh_coordinate_subsystem())
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.02]))

        # MELD setup
        isym = 1
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym, check_partials=True)
        ldxfer_builder.initialize(self.comm)

        # Scenario
        nonlinear_solver = om.NonlinearBlockGS(maxiter=50, iprint=2, use_aitken=False, rtol = 1E-14, atol=1E-14)
        linear_solver = om.LinearBlockGS(maxiter=50, iprint=2, use_aitken=False, rtol = 1e-14, atol=1e-14)
        self.mphys_add_scenario('cruise', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                 struct_builder=struct_builder,
                                                                 ldxfer_builder=ldxfer_builder),
                                                                 nonlinear_solver, linear_solver)

        for discipline in ['aero', 'struct']:
            self.mphys_connect_scenario_coordinate_source(
                'mesh_%s' % discipline, 'cruise', discipline)

        for dv in ['aoa', 'q_inf', 'vel', 'nu', 'mach', 'dv_struct']:
            self.connect(dv, f'cruise.{dv}')

# OpenMDAO setup

prob = om.Problem()
prob.model = Top()

prob.setup(force_alloc_complex=True, mode='rev')

om.n2(prob, show_browser=False, outfile='check_totals.html')

prob.run_model()
prob.check_totals(of=['cruise.C_L', 'cruise.ks_vmfailure', 'cruise.mass'], wrt=['aoa', 'dv_struct'], method='cs', step=1e-50)
#prob.check_totals(of=['cruise.C_L'], wrt=['aoa'], method='cs')
