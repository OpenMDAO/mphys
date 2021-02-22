#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys.multipoint import Multipoint

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_fun3d import Fun3dSfeBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_meld import MeldBuilder

from tacs import elements, constitutive, functions
from structural_patches_component import LumpPatches

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(om.Group):
    def setup(self):

        ################################################################################
        # FUN3D options
        ################################################################################
        meshfile = 'crm_invis_tet.b8.ugrid'
        boundary_tag_list = [3]
        aero_builder = Fun3dSfeBuilder(meshfile,boundary_tag_list)

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

        tacs_options = {
            'add_elements': add_elements,
            'mesh_file'   : 'CRM_box_2nd.bdf',
            'get_funcs'   : get_funcs
        }

        tacs_builder = TacsBuilder(tacs_options)

        ################################################################################
        # Transfer scheme options
        ################################################################################

        xfer_options = {
            'isym': 1,
            'n': 200,
            'beta': 0.5,
        }

        xfer_builder = MeldBuilder(xfer_options, aero_builder, tacs_builder)

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        lumper = self.add_subsystem('lumper',LumpPatches(N=240))

        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'mp_group',
            Multipoint(
                aero_builder   = aero_builder,
                struct_builder = tacs_builder,
                xfer_builder   = xfer_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

    def configure(self):
        self.dvs.add_output('aoa', val=0.0, units='deg')
        self.dvs.add_output('mach', val=0.2)
        self.dvs.add_output('reynolds_number', val=0.0)
        #self.dvs.add_output('q_inf', val=20000.0)
        self.dvs.add_output('q_inf', val=1000.0)
        self.dvs.add_output('ref_area', val=1.0)
        self.dvs.add_output('ref_length', val=1.0)
        self.dvs.add_output('moment_center',val=np.zeros(3))
        self.dvs.add_output('beta', val=0.0)

        # connect to the aero for each scenario
        self.connect('mach', ['mp_group.s0.solver_group.aero.flow.mach',
                              'mp_group.s0.solver_group.aero.forces.mach'])
        self.connect('aoa',['mp_group.s0.solver_group.aero.flow.aoa',
                              'mp_group.s0.aero_funcs.aoa'])
        self.connect('reynolds_number',['mp_group.s0.solver_group.aero.flow.reynolds_number',
                                        'mp_group.s0.solver_group.aero.forces.reynolds_number'])
        self.connect('q_inf',['mp_group.s0.solver_group.aero.forces.q_inf',
                              'mp_group.s0.aero_funcs.q_inf'])
        self.connect('ref_area',['mp_group.s0.aero_funcs.ref_area'])
        self.connect('ref_length',['mp_group.s0.aero_funcs.ref_length'])
        self.connect('moment_center',['mp_group.s0.aero_funcs.moment_center'])
        self.connect('beta',['mp_group.s0.aero_funcs.beta'])

        # add the structural thickness DVs
        ndv_struct = self.mp_group.struct_builder.get_ndv()
        self.dvs.add_output('thickness_lumped', val=0.01)
        self.connect('thickness_lumped', 'lumper.thickness_lumped')
        self.connect('lumper.thickness', ['mp_group.s0.solver_group.struct.dv_struct', 'mp_group.s0.struct_funcs.dv_struct',])


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup(mode='rev')
prob.final_setup()
#om.n2(prob, show_browser=False, outfile='mphys_as.html')
prob.run_model()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('Lift =',prob['mp_group.s0.aero_funcs.Lift'])
    print('CD =',prob['mp_group.s0.aero_funcs.C_D'])
    print('KS =',prob['mp_group.s0.struct_funcs.funcs.f_struct'])
output = prob.check_totals(of=['mp_group.s0.aero_funcs.Lift'], wrt=['thickness_lumped'],)
if MPI.COMM_WORLD.rank == 0:
    print('check_totals output',output)
