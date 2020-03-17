#rst Imports
from __future__ import print_function, division
import numpy as np
#from baseclasses import *
from mpi4py import MPI
from tacs import elements, constitutive, functions

from omfsi.as_multipoint import AS_Multipoint
from omfsi.vlm_component_configure import VLM_builder
# from omfsi.modal_structure_component import *
from omfsi.tacs_component_configure import TACS_builder
from omfsi.meld_xfer_component_configure import MELD_builder

import openmdao.api as om
# TODO we can change all of these imports to om.
from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce
from openmdao.api import NonlinearBlockGS, LinearBlockGS
from openmdao.api import view_model

use_modal = True
use_modal = False
comm = MPI.COMM_WORLD

class Top(om.Group):

    def setup(self):

        # VLM options

        aero_options = {
            'mesh_file':'wing_VLM.dat',
            'mach':0.85,
            'alpha':2*np.pi/180.,
            'q_inf':1000.,
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

        aero_options['N_nodes'], aero_options['N_elements'], aero_options['x_a0'], aero_options['quad'] = read_VLM_mesh(aero_options['mesh_file'])

        # VLM builder
        vlm_builder = VLM_builder(aero_options)

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
                    'mesh_file'   : 'wingbox_Y_Z_flip.bdf',
                    'f5_writer'   : f5_writer }

        # TACS builder
        if use_modal:
            tacs_setup['nmodes'] = 15
            tacs_builder = TACS_modal_builder(tacs_setup)
        else:
            tacs_builder = TACS_builder(tacs_setup)

        # MELD setup
        meld_options = {'isym': 1,
                        'n': 200,
                        'beta': 0.5}

        # MELD builder
        meld_builder = MELD_builder(meld_options, vlm_builder, tacs_builder)

        ################################################################################
        # MPHY setup
        ################################################################################
        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # each AS_Multipoint instance can keep multiple points with the same formulation
        mp = self.add_subsystem(
            'mp_group',
            AS_Multipoint(
                aero_builder   = vlm_builder,
                struct_builder = tacs_builder,
                xfer_builder   = meld_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphy_add_scenario('s0')

    def configure(self):

        # add AoA DV
        self.dvs.add_output('alpha', val=0.8)
        self.mp_group.s0.aero.mphy_add_aero_dv('alpha')
        self.connect('alpha', 'mp_group.s0.aero.alpha')

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

# optional but we can set it here.
model.nonlinear_solver = NonlinearRunOnce()
model.linear_solver = LinearRunOnce()

prob.setup()
om.n2(prob, show_browser=False, outfile='as_vlm_configure.html')

prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print('f_struct =',prob[scenario.name+'.struct_funcs.f_struct'])
    print('mass =',prob[scenario.name+'.struct_mass.mass'])
    print('cl =',prob[scenario.name+'.aero_funcs.CL_out'])
