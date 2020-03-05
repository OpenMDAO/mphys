#rst Imports
from __future__ import print_function, division
import numpy
from mpi4py import MPI

import openmdao.api as om

from omfsi.as_group import as_group

from baseclasses import *
from tacs import elements, constitutive

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(om.Group):

    def setup(self):
        # in the setup method, we add all components
        # we need the options to create the components, so we set them here

        ################################################################################
        # ADflow options
        ################################################################################
        aero_options = {
            # I/O Parameters
            'gridFile':'wing_vol.cgns',
            'outputDirectory':'.',
            'monitorvariables':['resrho','cl','cd'],
            # 'writeTecplotSurfaceSolution':False,
            # 'writevolumesolution':False,
            # 'writesurfacesolution':False,

            # Physics Parameters
            'equationType':'RANS',

            # Solver Parameters
            'smoother':'dadi',
            'CFL':1.5,
            'CFLCoarse':1.25,
            'MGCycle':'sg',
            'MGStartLevel':-1,
            'nCyclesCoarse':250,

            # ANK Solver Parameters
            'useANKSolver':True,
            # 'ankswitchtol':1e-1,
            'nsubiterturb': 5,

            # NK Solver Parameters
            'useNKSolver':True,
            'nkswitchtol':1e-4,

            # Termination Criteria
            'L2Convergence':1e-14,
            'L2ConvergenceCoarse':1e-2,
            'nCycles':10000,

            # force integration
            'forcesAsTractions':False,
        }

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

        tacs_setup = {'add_elements': add_elements,
                    'mesh_file'   : 'wingbox.bdf',
                    'get_funcs'   : get_funcs}

        ################################################################################
        # Transfer scheme setup
        ################################################################################
        meld_options = {'isym': 2,
                        'n': 200,
                        'beta': 0.5}

        # add the aerostructural group
        self.add_subsystem('as_group', as_group(aero_options     = aero_options,
                                                struct_options   = tacs_setup,
                                                transfer_options = meld_options,
                                                n_scenario       = 1))

    # def configure(self):
    #     # # set the solvers
    #     # self.nonlinear_solver = om.NonlinearRunOnce()
    #     # self.linear_solver = om.LinearRunOnce()

    #     # set the solver for the scenario group
    #     self.cruise1.nonlinear_solver = om.NonlinearRunOnce()
    #     self.cruise1.linear_solver = om.LinearRunOnce()

    #     self.cruise1.fsi_group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
    #     self.cruise1.fsi_group.linear_solver = LinearBlockGS(maxiter=100)
    #     self.cruise1.fsi_group.nonlinear_solver.options['iprint'] = 2

    #     # in the configure method, we have script-specific calls to add DVs, set flow conditions, etc.

    #     # move all of these inside respective components
    #     # Create AeroProblem
    #     ap = AeroProblem(name='wing',
    #         mach=0.8,
    #         altitude=10000,
    #         alpha=1.5,
    #         areaRef=45.5,
    #         chordRef=3.25,
    #         evalFuncs=['lift','drag', 'cl', 'cd']
    #     )
    #     ap.addDV('alpha',value=1.5,name='alpha')
    #     ap.addDV('mach',value=0.8,name='mach')
    #     self.dv.add_output('dv_struct',np.array(810*[0.01]))
    #     self.dv.add_output('alpha',np.array(1.5))
    #     self.dv.add_output('mach',np.array(0.8))

        # we can also add design variables, constraints and set the objective function here.
        # every solver is already initialized, so we can perform solver-specific calls
        # that are not in default OMFSI API.

# # before setup
# aero_assembler = AdflowAssembler(aero_options,ap)
# #aero_assembler.solver.addLiftDistribution(150, 'z')
# #aero_assembler.solver.addSlices('z', numpy.linspace(0.1, 14, 10))
# struct_assembler = TacsOmfsiAssembler(tacs_setup)
# xfer_assembler = MeldAssembler(meld_options,struct_assembler,aero_assembler)
# assembler = FsiAssembler(struct_assembler,aero_assembler,xfer_assembler)
# # after setup
# assembler.connection_srcs['dv_struct'] = 'dv.dv_struct'
# assembler.connection_srcs['alpha'] = 'dv.alpha'
# assembler.connection_srcs['mach'] = 'dv.mach'
# assembler.add_model_components(model)
# fsi_group = assembler.add_fsi_subsystem(model,scenario)

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile='as_configure.html')
prob.run_model()
# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print('cl =',prob[scenario.name+'.aero_funcs.cl'])
    print('cd =',prob[scenario.name+'.aero_funcs.cd'])
