# complex step partial derivative check of tacs
import numpy as np
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioStructural

from tacs import elements, constitutive, functions, TACS
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
    problem.addFunction('mass', functions.StructuralMass)
    problem.addFunction('ks_vmfailure', functions.KSFailure, safetyFactor=1.0, ksWeight=50.0)

    # Add random load vector
    f = fea_assembler.createVec()
    f[:] = np.random.rand(len(f))
    problem.addLoadToRHS(f)

class Top(Multipoint):

    def setup(self):
        tacs_builder = TacsBuilder(mesh_file='../input_files/debug.bdf', element_callback=element_callback,
                                   problem_setup=problem_setup, check_partials=True,
                                   write_solution=False)
        tacs_builder.initialize(self.comm)
        ndv_struct = tacs_builder.get_ndv()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))

        self.add_subsystem('mesh', tacs_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=tacs_builder))
        self.connect(f'mesh.{MPhysVariables.Structures.Mesh.COORDINATES}',
                     f'analysis.{MPhysVariables.Structures.COORDINATES}')
        self.connect('dv_struct', 'analysis.dv_struct')

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
        data = self.prob.check_partials(method='cs', step=1e-50, compact_print=True)
        for var, err in data.items():
            for out_var, in_var in err:
                with self.subTest(partial_pair=(out_var, in_var)):
                    # If fd check magnitude is exactly zero, use abs tol
                    if err[out_var, in_var]['magnitude'].fd == 0.0:
                        check_error = err[out_var, in_var]['abs error']
                    else:
                        check_error = err[out_var, in_var]['rel error']
                    assert_near_equal(check_error.reverse, 0.0, 1e-7)


if __name__ == '__main__':
    unittest.main()
