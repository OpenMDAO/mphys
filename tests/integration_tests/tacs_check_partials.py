# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np

import openmdao.api as om
from mphys.mphys_tacs import TacsBuilder
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


class Top(Multipoint):

    def setup(self):

        tacs_options = {'element_callback' : element_callback,
                        'mesh_file': '../input_files/debug.bdf'}

        tacs_builder = TacsBuilder(tacs_options, check_partials=True)
        tacs_builder.initialize(self.comm)
        ndv_struct = tacs_builder.get_ndv()
        dv_src_indices = tacs_builder.get_dv_src_indices()

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))

        self.add_subsystem('mesh', tacs_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_scenario('analysis', ScenarioStructural(struct_builder=tacs_builder))
        self.connect('mesh.x_struct0', 'analysis.x_struct0')
        self.connect('dv_struct', 'analysis.dv_struct', src_indices=dv_src_indices)

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        fea_solver = self.analysis.coupling.fea_solver

        # ==============================================================================
        # Setup structural problem
        # ==============================================================================
        # Structural problem
        sp = fea_solver.createStaticProblem(name='test')
        # Add TACS Functions
        sp.addFunction('mass', functions.StructuralMass)
        sp.addFunction('ks_vmfailure', functions.KSFailure, ksWeight=50.0)

        f = sp.F.getArray()
        f[:] = np.random.rand(len(f))

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.analysis.coupling.mphys_set_sp(sp)
        self.analysis.struct_post.mphys_set_sp(sp)


prob = om.Problem()
prob.model = Top()

prob.setup(mode='rev', force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs', compact_print=True)
