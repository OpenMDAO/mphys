"""
Mass minimization of uCRM wingbox subject to a constant vertical force

"""
from __future__ import division, print_function
import numpy as np

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import IndepVarComp, Group
from openmdao.api import NonlinearBlockGS, LinearBlockGS

from tacs import elements, constitutive, TACS, functions
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_multipoint import MPHYS_Multipoint

class Top(Group):

    def setup(self):

        ################################################################################
        # Tacs solver pieces
        ################################################################################
        def add_elements(mesh):
            rho = 2500.0  # density, kg/m^3
            E = 70.0e9 # elastic modulus, Pa
            nu = 0.3 # poisson's ratio
            kcorr = 5.0 / 6.0 # shear correction factor
            ys = 350e6  # yield stress, Pa
            thickness = 0.020
            min_thickness = 0.00
            max_thickness = 1.00

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

        def forcer_function(x_s,ndof):
            # apply uniform z load
            f_s = np.zeros(int(x_s.size/3)*ndof)
            f_s[2::ndof] = 100.0
            return f_s

        def f5_writer(tacs):
            flag = (TACS.ToFH5.NODES |
                    TACS.ToFH5.DISPLACEMENTS |
                    TACS.ToFH5.STRAINS |
                    TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
            f5.writeToFile('ucrm.f5')

        tacs_setup = {'add_elements': add_elements,
                    'mesh_file'   : 'CRM_box_2nd.bdf',
                    'get_funcs'   : get_funcs,
                    'load_function': forcer_function,
                    'f5_writer'   : f5_writer}

        # assembler = TacsOmfsiAssembler(tacs_setup,add_forcer=True)
        tacs_builder = TacsBuilder(tacs_setup)

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', IndepVarComp(), promotes=['*'])

        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'mp_group',
            MPHYS_Multipoint(struct_builder = tacs_builder)
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

    def configure(self):
        # add the structural thickness DVs
        ndv_struct = self.mp_group.struct_builder.get_ndv()
        self.dvs.add_output('dv_struct',np.array(240*[0.0031]))
        self.connect('dv_struct', ['mp_group.s0.struct.dv_struct'])

################################################################################
# OpenMDAO setup
################################################################################

prob = Problem()
prob.model = Top()
model = prob.model

model.add_design_var('dv_struct',lower=0.001,upper=0.075,scaler=1.0/1.0)
model.add_objective('mp_group.s0.struct.mass.mass',scaler=1.0/100000.0)
model.add_constraint('mp_group.s0.struct.funcs.f_struct',lower = 0.0, upper = 2.0/3.0,scaler=1000.0/1.0)

prob.driver = ScipyOptimizeDriver(debug_print=['objs','nl_cons'],maxiter=1500)
prob.driver.options['optimizer'] = 'SLSQP'

prob.setup()

prob.run_model()
# TODO verify that the optimization works
# prob.run_driver()
#for i in range(240):
#    print('final dvs',i,prob['dv_struct'][i])
