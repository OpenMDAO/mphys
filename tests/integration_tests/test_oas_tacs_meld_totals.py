# must compile funtofem and tacs in complex mode
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from openaerostruct.mphys import AeroBuilder
from pyfuntofem.mphys import MeldBuilder

from tacs import elements, constitutive, functions
from tacs.mphys import TacsBuilder

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

        # Create a dictionary to store options about the surface
        nx = ny = 3
        Lx = 1.0
        Ly = 2.0
        x = np.linspace(0, Lx, nx + 1)
        y = np.linspace(0, Ly, ny + 1)
        mesh = np.zeros([nx + 1, ny + 1, 3])
        mesh[:, :, 0], mesh[:, :, 1] = np.meshgrid(x, y, indexing='ij')

        surface = {
            # Wing definition
            "name": "wing",  # name of the surface
            "type": "aero",
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "projected",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "twist_cp": np.zeros(3),  # Define twist using 3 B-spline cp's
            # distributed along span
            "mesh": mesh,
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.0,  # CD of the surface at alpha=0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c": 0.12,  # thickness over chord ratio (NACA0015)
            "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
            # thickness
            "with_viscous": False,  # if true, compute viscous drag,
            "with_wave": False,
        }  # end of surface dictionary

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

        aero_builder = AeroBuilder([surface], options={"write_solution": False})
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero', aero_builder.get_mesh_coordinate_subsystem())

        # TACS
        tacs_options = {'element_callback' : element_callback,
                        'problem_setup': problem_setup,
                        'mesh_file': '../input_files/debug.bdf'}

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
        nonlinear_solver = om.NonlinearBlockGS(
            maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(
            maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        self.mphys_add_scenario('cruise', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                 struct_builder=struct_builder,
                                                                 ldxfer_builder=ldxfer_builder),
                                nonlinear_solver, linear_solver)

        for discipline in ['aero', 'struct']:
            self.mphys_connect_scenario_coordinate_source(
                'mesh_%s' % discipline, 'cruise', discipline)

        for dv in ['aoa', 'yaw', 'rho', 'mach', 'v', 'reynolds']:
            self.connect(dv, f'cruise.{dv}')
        self.connect('dv_struct', 'cruise.dv_struct')

class TestTACS(unittest.TestCase):
    N_PROCS=2
    def setUp(self):
        prob = om.Problem()
        prob.model = Top()

        prob.setup(mode='rev', force_alloc_complex=True)
        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    def test_derivatives(self):
        self.prob.run_model()
        print('----------------starting check totals--------------')
        data = self.prob.check_totals(of=['cruise.wing.CL', 'cruise.ks_vmfailure', 'cruise.mass'],
                                      wrt=['aoa', 'dv_struct'], method='fd', form='central',
                                      step=1e-5, step_calc='rel')
        assert_check_totals(data, atol=1e99, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()