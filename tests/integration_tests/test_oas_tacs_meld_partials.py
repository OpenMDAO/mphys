# must compile funtofem and tacs in complex mode
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioAeroStructural
from openaerostruct.mphys.aero_builder import AeroBuilder
from tacs.mphys import TacsBuilder
from funtofem import TransferScheme
from funtofem.mphys import MeldBuilder

from tacs import elements, constitutive, functions, TACS

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
        #mesh = mesh[:, ::-1, :]
        #mesh[:, :, 1] *= -1.0

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
        beta = 0.0
        vel = 178.
        re = 1e6

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK, val=aoa, units='deg')
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.YAW_ANGLE, val=beta, units='deg')
        dvs.add_output('rho', val=rho, units='kg/m**3')
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER, mach)
        dvs.add_output('v', vel, units='m/s')
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER, re, units="1/m")

        aero_builder = AeroBuilder([surface], options={"write_solution": False})
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero', aero_builder.get_mesh_coordinate_subsystem())

        # TACS setup
        struct_builder = TacsBuilder(mesh_file='../input_files/debug.bdf', element_callback=element_callback,
                                     problem_setup=problem_setup, check_partials=True,
                                     coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC],
                                     write_solution=False)

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

        self.connect(f'mesh_aero.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}',
                     f'cruise.{MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL}')
        self.connect(f'mesh_struct.{MPhysVariables.Structures.Mesh.COORDINATES}',
                     f'cruise.{MPhysVariables.Structures.COORDINATES}')

        for dv in [MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK,
                   MPhysVariables.Aerodynamics.FlowConditions.YAW_ANGLE,
                   MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER,
                   MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER,
                   'rho', 'v']:
            self.connect(dv, f'cruise.{dv}')
        self.connect('dv_struct', 'cruise.dv_struct')


class TestOAS(unittest.TestCase):
    N_PROCS=2
    def setUp(self):
        prob = om.Problem()
        prob.model = Top()

        prob.setup(mode='rev', force_alloc_complex=True)
        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    @unittest.skipUnless(TACS.dtype == complex and TransferScheme.dtype == complex,
                         "TACS/FunToFem must be compiled in complex mode.")
    def test_derivatives(self):
        self.prob.run_model()
        print('----------------starting check totals--------------')
        data = self.prob.check_partials(method='cs', step=1e-50, compact_print=True)
        for comp in data:
            with self.subTest(component=comp):
                err = data[comp]
                for out_var, in_var in err:
                    with self.subTest(partial_pair=(out_var, in_var)):
                        # If fd check magnitude is exactly zero, use abs tol
                        if err[out_var, in_var]['magnitude'].fd == 0.0:
                            check_error = err[out_var, in_var]['abs error']
                        else:
                            check_error = err[out_var, in_var]['rel error']
                        if check_error.reverse is None:
                            assert_near_equal(check_error.forward, 0.0, 1e-5)
                        else:
                            assert_near_equal(check_error.reverse, 0.0, 1e-5)


if __name__ == '__main__':
    unittest.main()
