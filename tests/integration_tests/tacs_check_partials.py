# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np

import openmdao.api as om
from mphys.solver_builders.mphys_tacs import TacsBuilder
from mphys.multipoint import Multipoint
from mphys.scenario_structural import ScenarioStructural

from tacs import elements, constitutive, functions

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

        tacs_options = {'element_callback' : element_callback,
                        'problem_setup': problem_setup,
                        'mesh_file': '../input_files/debug.bdf'}

        tacs_builder = TacsBuilder(tacs_options, check_partials=True, coupled=False, write_solution=False)
        tacs_builder.initialize(self.comm)
        ndv_struct = tacs_builder.get_ndv()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))

        self.add_subsystem('mesh', tacs_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=tacs_builder))
        self.connect('mesh.x_struct0', 'analysis.x_struct0')
        self.connect('dv_struct', 'analysis.dv_struct')


prob = om.Problem()
prob.model = Top()

prob.setup(mode='rev', force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs', compact_print=True)
