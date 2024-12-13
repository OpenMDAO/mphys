import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioAerodynamic
from openaerostruct.mphys import AeroBuilder


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
            "name": "Wing",
            "mesh": mesh,
            "symmetry": True,
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

        mach = 0.85
        aoa = 1.0
        rho = 1.2
        vel = 178.0
        re = 1e6

        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK, val=aoa, units="deg")
        dvs.add_output("rho", val=rho, units="kg/m**3")
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER, mach)
        dvs.add_output("v", vel, units="m/s")
        dvs.add_output(MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER, re, units="1/m")

        aero_builder = AeroBuilder([surface], options={"write_solution": False})
        aero_builder.initialize(self.comm)

        self.add_subsystem('mesh', aero_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('cruise', ScenarioAerodynamic(aero_builder=aero_builder))
        self.connect(f'mesh.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}',
                     f'cruise.{MPhysVariables.Aerodynamics.Surface.COORDINATES}')

        for dv in [MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK,
                   MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER,
                   MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER,
                   'rho', 'v']:
            self.connect(dv, f'cruise.{dv}')

class TestOAS(unittest.TestCase):
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
