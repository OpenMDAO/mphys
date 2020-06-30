import numpy as np

import openmdao.api as om
from mphys.base_classes import CoupledAnalysis
from mphys.mphys_adflow import ADflow_builder
from mphys.mphys_adflow import ADflowGroup
from mphys.mphys_tacs import TACSGroup
from tacs import elements, constitutive, TACS, functions

from baseclasses import *
from mpi4py import MPI

class Top(om.Group):

    def setup(self):
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        dvs.add_output('alpha', val=1.5)
        self.connect('alpha', ['aerostruct.alpha'])

        aero_options = {
            # I/O Parameters
            'gridFile':'wing_vol.cgns',
            'outputDirectory':'.',
            'monitorvariables':['resrho','cl','cd'],
            'writeTecplotSurfaceSolution':True,

            # Physics Parameters
            'equationType':'RANS',

            # Solver Parameters
            'smoother':'dadi',
            'CFL':0.5,
            'CFLCoarse':0.25,
            'MGCycle':'sg',
            'MGStartLevel':-1,
            'nCyclesCoarse':250,

            # ANK Solver Parameters
            'useANKSolver':True,
            'nsubiterturb': 5,

            # NK Solver Parameters
            'useNKSolver':True,
            'nkswitchtol':1e-4,

            # Termination Criteria
            'L2Convergence':1e-2,
            'L2ConvergenceCoarse':1e-2,
            'nCycles':1000,
        }

        
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

        solver_options = {'add_elements': add_elements,
                    'mesh_file'   : 'wingbox.bdf',
                    'get_funcs'   : get_funcs,
                    'load_function': forcer_function,
                    'f5_writer'   : f5_writer}
        # tacs_builder = TACS_builder(tacs_options)


        ap0 = AeroProblem(
            name='ap0',
            mach=0.8,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['cl','cd']
        )
        ap0.addDV('alpha',value=1.5,name='alpha')
        
        # the solver must be created before added the group as a subsystem.
        MDA =CoupledAnalysis()
        MDA.add_subsystem('aero',
                          ADflowGroup(aero_problem = ap0, 
                           solver_options = aero_options, 
                           group_options = {
                               'geo_disp': True,
                               'mesh': False,
                               'deformer': True,
                               'forces': True
                           }),
                           promotes=['alpha', 'u_a', 'f_a'])
        
        MDA.add_subsystem('dixp_xfer',
                          ADflowGroup(aero_problem = ap0, 
                           solver_options = aero_options, 
                           group_options = {
                               'geo_disp': True,
                               'mesh': False,
                               'deformer': True,
                               'forces': True
                           }),
                           promotes=['alpha', 'u_a', 'f_a'])


        MDA.add_subsystem('struct',
                          TACSGroup(solver_options=solver_options,
                                              group_options={
                                                  'loads':False,
                                              }),
                           promotes=['alpha', 'u_s', 'f_s'])
                           

        self.add_subsystem('aerostruct', MDA)






################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()

prob.setup()
om.n2(prob, show_browser=False, outfile='mphys_aerostruct.html')

# prob.run_model()

# prob.model.list_inputs(units=True)
# prob.model.list_outputs(units=True)

# # prob.model.list_outputs()
# if MPI.COMM_WORLD.rank == 0:
#     print("Scenario 0")
#     print('lo cl =', prob.get_val('multi.lo_mach.cl', get_remote=True))
#     print('hi cl =', prob.get_val('multi.hi_mach.cl', get_remote=True))
