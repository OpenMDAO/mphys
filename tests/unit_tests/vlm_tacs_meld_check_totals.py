from __future__ import print_function, division
from mpi4py import MPI
import numpy as np

import openmdao.api as om

from mphys.multipoint import Multipoint
from mphys.mphys_vlm import VlmBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_meld import MeldBuilder

from tacs import elements, constitutive, functions, TACS

use_modal = False

class Top(om.Group):
    def setup(self):
        # VLM options
        aero_options = {
            'mesh_file':'debug_VLM.dat',
            'mach':0.85,
            'aoa':1*np.pi/180.,
            'q_inf':25000.,
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

        aero_options['N_nodes'], aero_options['N_elements'], aero_options['x_aero0'], aero_options['quad'] = read_VLM_mesh(aero_options['mesh_file'])
        self.aero_options = aero_options

        # VLM builder
        aero_builder = VlmBuilder(aero_options)

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
        tacs_setup = {'add_elements': add_elements,
                      'get_funcs'   : get_funcs,
                      'mesh_file'   : 'debug.bdf',
                      'f5_writer'   : f5_writer }

        # TACS assembler

        if use_modal:
            tacs_setup['nmodes'] = 15
            #struct_assembler = ModalStructAssembler(tacs_setup)
        else:
            struct_builder = TacsBuilder(tacs_setup,check_partials=True)

        # MELD setup

        meld_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}

        xfer_builder = MeldBuilder(meld_options,aero_builder,struct_builder,check_partials=True)

        # Multipoint group
        dvs = self.add_subsystem('dvs',om.IndepVarComp(), promotes=['*'])

        mp = self.add_subsystem(
            'mp_group',
            Multipoint(aero_builder = aero_builder,
                       struct_builder = struct_builder,
                       xfer_builder = xfer_builder)
        )
        s0 = mp.mphys_add_scenario('s0')

    def configure(self):
        self.dvs.add_output('aoa', self.aero_options['aoa'], units='rad')
        self.connect('aoa',['mp_group.s0.solver_group.aero.aoa'])

        self.dvs.add_output('dv_struct',np.array([0.03]))
        self.connect('dv_struct',['mp_group.s0.solver_group.struct.dv_struct, mp_group.s0.struct_funcs.dv_struct'])

# OpenMDAO setup

prob = om.Problem()
prob.model = Top()

prob.setup(force_alloc_complex=True)

prob.model.mp_group.s0.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, iprint=2, use_aitken=True, atol=1E-9)
prob.model.mp_group.s0.linear_solver = om.LinearBlockGS(maxiter=10, iprint=2)

prob.setup(force_alloc_complex=True, mode='rev')

prob.run_model()
prob.check_totals(of=['mp_group.s0.struct.funcs.func_struct'], wrt=['aoa'], method='cs')
