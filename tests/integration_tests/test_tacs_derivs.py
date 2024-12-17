"""
@File    :   test_aero_derivs.py
@Time    :   2020/12/17
@Author  :   Josh Anibal
@Desc    :   File for testing the derivatives of the mphys adflow wrapper
"""

# === Standard Python modules ===
import unittest
import os

# === External Python modules ===
import numpy as np

# === Extension modules ===
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals

from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioStructural

from tacs import elements, constitutive, functions, TACS
from tacs.mphys import TacsBuilder

baseDir = os.path.dirname(os.path.abspath(__file__))

# Callback function used to setup TACS element objects and DVs
def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    rho = 2780.0  # density, kg/m^3
    E = 73.1e9  # elastic modulus, Pa
    nu = 0.33  # poisson's ratio
    ys = 324.0e6  # yield stress, Pa
    thickness = 0.012
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
    # Set convergence to be tight for test
    problem.setOption('L2Convergence', 1e-20)
    problem.setOption('L2ConvergenceRel', 1e-20)

    # Add TACS Functions
    problem.addFunction('mass', functions.StructuralMass)
    problem.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

    # Add gravity load
    g = np.array([0.0, 0.0, -9.81])  # m/s^2
    problem.addInertialLoad(g)

class Top(Multipoint):

    def setup(self):
        tacs_builder = TacsBuilder(mesh_file='../input_files/debug.bdf', element_callback=element_callback,
                                   problem_setup=problem_setup,
                                   check_partials=True,
                                   coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC],
                                   write_solution=False)
        tacs_builder.initialize(self.comm)
        ndv_struct = tacs_builder.get_ndv()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))

        f_size = tacs_builder.get_ndof() * tacs_builder.get_number_of_nodes()
        forces = self.add_subsystem('forces', om.IndepVarComp(), promotes=['*'])
        forces.add_output('f_struct', val=np.ones(f_size), distributed=True)

        self.add_subsystem('mesh', tacs_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=tacs_builder))

        self.connect(f'mesh.{MPhysVariables.Structures.Mesh.COORDINATES}',
                     f'analysis.{MPhysVariables.Structures.COORDINATES}')
        self.connect('dv_struct', 'analysis.dv_struct')
        self.connect('f_struct', f'analysis.{MPhysVariables.Structures.Loads.AERODYNAMIC}')


class TestTACS(unittest.TestCase):
    N_PROCS=2
    def setUp(self):
        prob = om.Problem()
        prob.model = Top()

        prob.setup(mode='rev', force_alloc_complex=True)
        self.prob = prob

    def test_run_model(self):
        self.prob.run_model()

    @unittest.skipUnless(TACS.dtype == complex, "TACS must be compiled in complex mode.")
    def test_derivatives(self):
        self.prob.run_model()
        print('----------------starting check totals--------------')
        data = self.prob.check_totals(of=['analysis.ks_vmfailure', 'analysis.mass'],
                                      wrt=f'mesh.fea_mesh.{MPhysVariables.Structures.Mesh.COORDINATES}', method='cs',
                                      step=1e-50, step_calc='rel')
        assert_check_totals(data, atol=1e99, rtol=1e-7)

        data = self.prob.check_totals(of=['analysis.ks_vmfailure'], wrt='f_struct',
                                      method='cs', step=1e-50, step_calc='rel')
        assert_check_totals(data, atol=1e99, rtol=5e-8)

        data = self.prob.check_totals(of=['analysis.ks_vmfailure', 'analysis.mass'],
                                      wrt='dv_struct', method='cs',
                                      step=1e-50, step_calc='rel')
        assert_check_totals(data, atol=1e99, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
