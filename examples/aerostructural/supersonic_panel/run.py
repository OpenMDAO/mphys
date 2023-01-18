import numpy as np
from mpi4py import MPI
import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural

from structures_mphys import StructBuilder
from aerodynamics_mphys import AeroBuilder
from xfer_mphys import XferBuilder 

from geometry_morph import GeometryBuilder

comm = MPI.COMM_WORLD
rank = comm.rank

# panel geometry
panel_chord = 0.3
panel_width = 0.01

# panel discretization
N_el_struct = 20
N_el_aero = 7

# Mphys
class Model(Multipoint):
    def setup(self):

        # ivc
        self.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        self.ivc.add_output('modulus', val=70E9)
        self.ivc.add_output('yield_stress', val=270E6)
        self.ivc.add_output('density', val=2800.)
        self.ivc.add_output('mach', val=5.)
        self.ivc.add_output('qdyn', val=3E4)
        self.ivc.add_output('aoa', val=3., units='deg')
        self.ivc.add_output('geometry_morph_param', val=1.)

        # create dv_struct, which is the thickness of each structural element
        thickness = 0.001*np.ones(N_el_struct)
        self.ivc.add_output('dv_struct', thickness)

        # structure setup and builder
        structure_setup = {'panel_chord'  : panel_chord,
                           'panel_width'  : panel_width,
                           'N_el'         : N_el_struct}

        struct_builder = StructBuilder(structure_setup)
        struct_builder.initialize(self.comm)

        # aero builder
        aero_setup = {'panel_chord'  : panel_chord,
                      'panel_width'  : panel_width,
                      'N_el'         : N_el_aero}

        aero_builder = AeroBuilder(aero_setup)
        aero_builder.initialize(self.comm)

        # xfer builder
        xfer_builder = XferBuilder(
            aero_builder=aero_builder,
            struct_builder=struct_builder
        )
        xfer_builder.initialize(self.comm)

        # geometry
        builders = {'struct': struct_builder, 'aero': aero_builder}
        geometry_builder = GeometryBuilder(builders)

        self.add_subsystem('struct_mesh', struct_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem('aero_mesh', aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem('geometry', geometry_builder.get_mesh_coordinate_subsystem(), promotes=['*'])

        self.connect('struct_mesh.x_struct0', 'x_struct0_in')
        self.connect('aero_mesh.x_aero0', 'x_aero0_in')
 
        # aerostructural analysis
        nonlinear_solver = om.NonlinearBlockGS(maxiter=100, iprint=2, use_aitken=True, aitken_initial_factor=0.5)
        linear_solver = om.LinearBlockGS(maxiter=40, iprint=2, use_aitken=True, aitken_initial_factor=0.5)
        self.mphys_add_scenario('aerostructural',
                                ScenarioAeroStructural(
                                    aero_builder=aero_builder, struct_builder=struct_builder,
                                    ldxfer_builder=xfer_builder),
                                coupling_nonlinear_solver=nonlinear_solver,
                                coupling_linear_solver=linear_solver)

        for var in ['modulus', 'yield_stress', 'density', 'mach', 'qdyn', 'aoa', 'dv_struct', 'x_struct0', 'x_aero0']:
            self.connect(var, 'aerostructural.'+var)

# run model and check derivatives
if __name__ == "__main__":

    prob = om.Problem()
    prob.model = Model()
    prob.setup(mode='rev')

    om.n2(prob, show_browser=False, outfile='n2.html')
    
    prob.run_model()
    print('mass =        ' + str(prob['aerostructural.mass']))
    print('func_struct = ' + str(prob['aerostructural.func_struct']))
    print('C_L =         ' + str(prob['aerostructural.C_L']))
    
    prob.check_totals(
        of=['aerostructural.mass',
            'aerostructural.func_struct',
            'aerostructural.C_L'], 
        wrt=['modulus',
             'yield_stress',
             'density',
             'mach',
             'qdyn',
             'aoa', 
             'dv_struct',
             'geometry_morph_param'],  
        step_calc='rel_avg',
        compact_print=True
    )

    prob.check_partials(compact_print=True, step_calc='rel_avg')
