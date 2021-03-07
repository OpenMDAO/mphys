import numpy as np
import argparse

from mpi4py import MPI

import openmdao.api as om

from tacs import elements, constitutive, functions, TACS

from mphys.multipoint import Multipoint
from mphys.mphys_vlm import VlmBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_modal_solver import ModalBuilder
from mphys.mphys_meld import MeldBuilder


parser=argparse.ArgumentParser()
parser.add_argument('--modal', default=False, action="store_true")
args = parser.parse_args()

class Top(om.Group):

    def setup(self):
        self.modal_struct = args.modal

        # VLM options
        self.aero_options = {
            'mesh_file':'wing_VLM.dat',
            'mach':0.85,
            'alpha':2*np.pi/180.,
            'q_inf':3000.,
            'vel':178.,
            'mu':3.5E-5,
        }

        # VLM mesh read

        def read_VLM_mesh(mesh):
            f=open(mesh, "r")
            contents = f.read().split()

            a = [i for i in contents if 'NODES' in i][0]
            N_nodes = int(a[a.find("=")+1:a.find(",")])
            a = [i for i in contents if 'ELEMENTS' in i][0]
            N_elements = int(a[a.find("=")+1:a.find(",")])

            a = np.array(contents[16:16+N_nodes*3],'float')
            X = a[0:N_nodes*3:3]
            Y = a[1:N_nodes*3:3]
            Z = a[2:N_nodes*3:3]
            a = np.array(contents[16+N_nodes*3:None],'int')
            quad = np.reshape(a,[N_elements,4])

            xa = np.c_[X,Y,Z].flatten(order='C')

            f.close()

            return N_nodes, N_elements, xa, quad

        self.aero_options['N_nodes'], self.aero_options['N_elements'], self.aero_options['x_aero0'], self.aero_options['quad'] = read_VLM_mesh(self.aero_options['mesh_file'])

        # VLM builder
        vlm_builder = VlmBuilder(self.aero_options)

        # TACS setup

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

        def f5_writer(tacs):
            flag = (TACS.ToFH5.NODES |
                    TACS.ToFH5.DISPLACEMENTS |
                    TACS.ToFH5.STRAINS |
                    TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile('wingbox.f5')


        # common setup options
        tacs_setup = {
            'add_elements': add_elements,
            'get_funcs'   : get_funcs,
            'mesh_file'   : 'wingbox_Y_Z_flip.bdf',
            # 'f5_writer'   : f5_writer,
        }

        if self.modal_struct:
            nmodes = 15
            struct_builder = ModalBuilder(tacs_setup,nmodes)
        else:
            struct_builder = TacsBuilder(tacs_setup)

        # MELD setup
        meld_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}

        # MELD builder
        meld_builder = MeldBuilder(meld_options, vlm_builder, struct_builder)

        ################################################################################
        # MPHYS setup
        ################################################################################
        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # each MPHYS_Multipoint instance can keep multiple points with the same formulation
        mp = self.add_subsystem(
            'mp_group',
            Multipoint(
                aero_builder   = vlm_builder,
                struct_builder = struct_builder,
                xfer_builder   = meld_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

    def configure(self):

        # add aero DVs
        for dv_name in ['alpha','q_inf','vel','mu','mach']:
            if dv_name == 'alpha':
                self.dvs.add_output(dv_name, val=self.aero_options[dv_name], units='rad')
            else:
                self.dvs.add_output(dv_name, val=self.aero_options[dv_name])
            self.connect(dv_name, 'mp_group.s0.solver_group.aero.%s' % dv_name)

        # add the structural thickness DVs
        ndv_struct = self.mp_group.struct_builder.get_ndv()
        self.dvs.add_output('dv_struct', np.array(ndv_struct*[0.002]))

        # connect solver data
        if self.modal_struct:
            self.connect('dv_struct', ['mp_group.struct_mesh.dv_struct'])
        else:
            self.connect('dv_struct', ['mp_group.s0.solver_group.struct.dv_struct', 'mp_group.s0.struct_funcs.dv_struct'])

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model

# optional but we can set it here.
model.nonlinear_solver = om.NonlinearRunOnce()
model.linear_solver = om.LinearRunOnce()


prob.setup()

# model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=20, iprint=2, use_aitken=False, rtol = 1E-14, atol=1E-14)
# model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=20, iprint=2, rtol = 1e-14, atol=1e-14)

if args.modal:
    n2name = 'mphys_as_vlm_modal_meld.html'
else:
    n2name = 'mphys_as_vlm_tacs_meld.html'

om.n2(prob, show_browser=False, outfile=n2name)

prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print('cl =',prob['mp_group.s0.solver_group.aero.forces.CL'])
    if not args.modal:
        print('f_struct =',prob['mp_group.s0.struct_funcs.funcs.f_struct'])
        print('mass =',prob['mp_group.s0.struct_funcs.mass.mass'])
