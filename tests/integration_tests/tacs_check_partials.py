# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np

import openmdao.api as om
from mphys.mphys_tacs import TacsBuilder
from mphys.multipoint import Multipoint

from tacs import elements, constitutive, functions


class Top(om.Group):

    def setup(self):
        def add_elements(mesh):
            rho = 2780.0            # density, kg/m^3
            E = 73.1e9              # elastic modulus, Pa
            nu = 0.33               # poisson's ratio
            kcorr = 5.0 / 6.0       # shear correction factor
            ys = 324.0e6            # yield stress, Pa
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
        def forcer(x_s0,ndof):
            return np.random.rand(int(x_s0.size/3*ndof))

        tacs_options = {'add_elements': add_elements,
                        'get_funcs'   : get_funcs,
                        'mesh_file'   : 'debug.bdf',
                        'load_function' : forcer}

        tacs_builder = TacsBuilder(tacs_options, check_partials=True)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'mp_group',
            Multipoint(struct_builder = tacs_builder)
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

    def configure(self):
        # add the structural thickness DVs
        ndv_struct = self.mp_group.struct_builder.get_ndv()
        self.dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))
        self.connect('dv_struct', ['mp_group.s0.solver_group.struct.dv_struct', 'mp_group.s0.struct_funcs.dv_struct'])

prob = om.Problem()
prob.model = Top()

prob.setup(mode='rev',force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs',compact_print=True)
