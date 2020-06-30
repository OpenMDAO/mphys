#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys.multipoint import Multipoint

# these imports will be from the respective codes' repos rather than omfsi
from mphys.mphys_tacs import TacsBuilder
from tacs import elements, constitutive, functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(om.Group):

    def setup(self):

        ################################################################################
        # TACS options
        ################################################################################
        def add_elements(mesh):
            rho = 2780.0            # density, kg/m^3
            E = 73.1e9              # elastic modulus, Pa
            nu = 0.33               # poisson's ratio
            kcorr = 5.0 / 6.0       # shear correction factor
            ys = 324.0e6            # yield stress, Pa
            thickness= 0.020
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

        def load_function(x_s, ndof):
            f_s = np.zeros(int(x_s.size/3)*ndof)
            f_s[2::ndof] = 100.0
            return f_s

        tacs_options = {
            'add_elements': add_elements,
            'mesh_file'   : 'wingbox.bdf',
            'get_funcs'   : get_funcs,
            'load_function':load_function
        }

        tacs_builder = TacsBuilder(tacs_options)

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
        self.connect('dv_struct', ['mp_group.s0.struct.dv_struct'])

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile='mphys_struct.html')
prob.run_model()
# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")

