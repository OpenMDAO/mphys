#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.mphys_meld import MeldBuilder
from mphys.mphys_vlm import VlmBuilder
from pyshell.mphys import PyshellBuilder
from pyshell import ReadBdf

from dv_map import dv_map

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

use_fun3d = True

class Top(Multipoint):
    def setup(self):
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        aoa = 0.0
        mach = 0.85
        q_inf = 120.0
        vel = 217.6
        nu = 1.4E-5

        mesh_file = 'CRM_VLM_mesh_extended.dat'
        aero_builder = VlmBuilder(mesh_file, complex_step=False)
        aero_builder.initialize(self.comm)

        dvs.add_output('aoa', val=aoa, units='deg')
        dvs.add_output('mach', mach)
        dvs.add_output('q_inf', q_inf)
        dvs.add_output('vel', vel)
        dvs.add_output('nu', nu)
        aero_dvs = ['aoa','mach','q_inf','vel','nu']

        # pyshell options
        pyshell_options = {
            'mesh_file'   : 'CRM_box_2nd.bdf',
            'dv_map'      : dv_map,
            'ndv_per_el'  : 1,
            'load_list'   : ['f_struct']
        }

        bdf = ReadBdf('CRM_box_2nd.bdf', comm = self.comm)
        bdf.read_nodes()
        bdf.read_elements()
        bdf.read_bc()
        dv_struct = 0.01*np.ones(np.max(bdf.dv_struct_mapping)+1)

        struct_builder = PyshellBuilder(pyshell_options)
        struct_builder.initialize(self.comm)
        ndv_struct = dv_struct.size

        dvs.add_output('dv_struct', val=dv_struct)

        # Transfer scheme options
        isym = 1
        ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym)
        ldxfer_builder.initialize(self.comm)

        self.add_subsystem('mesh_aero',aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem('mesh_struct',struct_builder.get_mesh_coordinate_subsystem())

        nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-14, atol=1e-14)
        #linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol = 1e-8, atol=1e-8)
        linear_solver = om.PETScKrylov(maxiter=1, iprint=2, atol=1e-14, rtol=1e-14)
        #linear_solver = om.ScipyKrylov(maxiter=1, iprint=2)
        linear_solver.precon = om.LinearBlockGS(maxiter = 3)
        self.mphys_add_scenario('cruise',ScenarioAeroStructural(aero_builder=aero_builder,
                                                                struct_builder=struct_builder,
                                                                ldxfer_builder=ldxfer_builder),
                                         nonlinear_solver, linear_solver)

        for discipline in ['aero','struct']:
            self.mphys_connect_scenario_coordinate_source('mesh_%s' % discipline, 'cruise', discipline)

        self.connect('dv_struct', ['cruise.dv_struct'])
        for dv in aero_dvs:
            self.connect(dv, f'cruise.{dv}')


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
prob.setup(mode='rev', force_alloc_complex=True)
#prob.final_setup()
#prob.set_complex_step_mode(True)
#prob['aoa'] = 0.0 + 1.0e-30j

om.n2(prob, show_browser=False, outfile='crm_aerostruct.html')

prob.run_model()
prob.check_totals(of='cruise.func_struct', wrt='aoa')
