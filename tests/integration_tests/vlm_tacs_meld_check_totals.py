from __future__ import print_function, division
from mpi4py import MPI
import numpy as np

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.mphys_vlm import VlmBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_meld import MeldBuilder

from tacs import elements, constitutive, functions, TACS

use_modal = False

def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness = 0.003
    min_thickness = 0.002
    max_thickness = 0.05

    num_components = mesh.getNumComponents()
    for i in range(num_components):
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                     min_thickness, max_thickness)
        element = None
        if descript in ['CQUAD', 'CQUADR', 'CQUAD4']:
            element = elements.MITCShell(2, stiff, component_num=i)
        mesh.setElement(i, element)

    ndof = 6
    ndv = num_components

    return ndof, ndv


def get_funcs(tacs):
    ks_weight = 50.0
    return [functions.KSFailure(tacs, ks_weight), functions.StructuralMass(tacs)]


def f5_writer(tacs):
    flag = (TACS.ToFH5.NODES |
            TACS.ToFH5.DISPLACEMENTS |
            TACS.ToFH5.STRAINS |
            TACS.ToFH5.EXTRAS)
    f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
    f5.writeToFile('wingbox.f5')


class Top(Multipoint):
    def setup(self):
        # VLM
        mesh_file = '../input_files/debug_VLM.dat'
        mach = 0.85,
        aoa = 1.0
        q_inf = 25000.
        vel = 178.
        nu = 3.5E-5

        aero_builder = VlmBuilder(mesh_file, complex_step=True)
        aero_builder.initialize(self.comm)

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('nu', nu)

        self.add_subsystem('mesh_aero', aero_builder.get_mesh_coordinate_subsystem())

        # TACS
        tacs_options = {'add_elements': add_elements,
                        'get_funcs': get_funcs,
                        'mesh_file': '../input_files/debug.bdf',
                        'f5_writer': f5_writer}

        if use_modal:
            tacs_options['nmodes'] = 15
            #struct_assembler = ModalStructAssembler(tacs_options)
        else:
            struct_builder = TacsBuilder(tacs_options, check_partials=True)

        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem('mesh_struct', struct_builder.get_mesh_coordinate_subsystem())
        dvs.add_output('dv_struct', np.array(ndv_struct*[0.02]))

        # MELD setup
        isym = 1
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym, check_partials=True)
        ldxfer_builder.initialize(self.comm)

        # Scenario

        #NOTE: use_aitken creates issues with complex step in check_totals
        nonlinear_solver = om.NonlinearBlockGS(
            maxiter=200, iprint=2, use_aitken=False, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(
            maxiter=200, iprint=2, use_aitken=False, rtol=1e-14, atol=1e-14)
        self.mphys_add_scenario('cruise', ScenarioAeroStructural(aero_builder=aero_builder,
                                                                 struct_builder=struct_builder,
                                                                 ldxfer_builder=ldxfer_builder),
                                nonlinear_solver, linear_solver)

        for discipline in ['aero', 'struct']:
            self.mphys_connect_scenario_coordinate_source(
                'mesh_%s' % discipline, 'cruise', discipline)

        for dv in ['aoa', 'q_inf', 'vel', 'nu', 'mach', 'dv_struct']:
            self.connect(dv, f'cruise.{dv}')

# OpenMDAO setup

prob = om.Problem()
prob.model = Top()

prob.setup(force_alloc_complex=True, mode='rev')

om.n2(prob, show_browser=False, outfile='check_totals.html')

prob.run_model()
prob.check_totals(of=['cruise.func_struct'], wrt=['aoa'], method='cs')
#prob.check_totals(of=['cruise.C_L'], wrt=['aoa'], method='cs')
