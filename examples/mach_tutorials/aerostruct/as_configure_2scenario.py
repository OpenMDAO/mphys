#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from omfsi.as_multipoint import AS_Multipoint

# these imports will be from the respective codes' repos rather than omfsi
from omfsi.adflow_component_configure import ADflow_builder
from omfsi.tacs_component_configure import TACS_builder
from omfsi.meld_xfer_component_configure import MELD_builder

from baseclasses import *
from tacs import elements, constitutive, functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(om.Group):

    def setup(self):

        ################################################################################
        # ADflow options
        ################################################################################
        aero_options = {
            # I/O Parameters
            'gridFile':'wing_vol.cgns',
            'outputDirectory':'.',
            'monitorvariables':['resrho','cl','cd'],
            'writeTecplotSurfaceSolution':False,
            'writevolumesolution':False,
            'writesurfacesolution':False,

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

        adflow_builder = ADflow_builder(aero_options)

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
            'mesh_file'   : 'wingbox.bdf',
            'get_funcs'   : get_funcs
        }

        tacs_builder = TACS_builder(tacs_options)

        ################################################################################
        # Transfer scheme options
        ################################################################################
        xfer_options = {
            'isym': 2,
            'n': 200,
            'beta': 0.5,
        }

        meld_builder = MELD_builder(xfer_options, adflow_builder, tacs_builder)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('foo')

        # each AS_Multipoint instance can keep multiple points with the SAME FORMULATION
        # e.g. these cases will have the same aero struct and xfer formulation and meshes
        # solver-specific options can be different, and they can be adjusted in configure.
        mp = self.add_subsystem(
            'mp_group',
            # this AS_Multipoint instance uses ADflow, TACS and MELD. This mp_group
            # can contain multiple points (scenarios, flow conditions, etc.); however,
            # all of these cases must use the same numerical formulation. If the user
            # wants to add additional points with a different numerical formulation,
            # they need to create another instance of AS_Multipoint with desired
            # builders.
            AS_Multipoint(
                aero_builder   = adflow_builder,
                struct_builder = tacs_builder,
                xfer_builder   = meld_builder
            ),
            # the user can define a custom limit on proc count for this group of
            # multipoint cases here
            max_procs=MPI.COMM_WORLD.size
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphy_add_scenario(
            # name of the point
            's0',
            # The users can specify the proc counts here using an API very similar
            # to the default OpenMDAO API (Note this is not a default OpenMDAO call)
            min_procs=1,
            max_procs=MPI.COMM_WORLD.size,
            # scenario kwargs will overload any kwargs defaults from the MP group,
            # useful if you wanted to customize this point
            aero_kwargs={},
            struct_kwargs={},
            xfer_kwargs={},
        )

        # similarly, add a second point. the optional arguments above are all defaults
        mp.mphy_add_scenario('s1')

    def configure(self):
        return
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.
        ap0 = AeroProblem(
            name='ap0',
            mach=0.8,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['lift','drag', 'cl', 'cd']
        )
        ap0.addDV('alpha',value=1.5,name='alpha')
        ap0.addDV('mach',value=0.8,name='mach')

        # similarly, add the aero problem for the second analysis point
        ap1 = AeroProblem(
            name='ap1',
            mach=0.7,
            altitude=10000,
            alpha=1.5,
            areaRef=45.5,
            chordRef=3.25,
            evalFuncs=['lift','drag', 'cl', 'cd']
        )
        ap1.addDV('alpha',value=1.5,name='alpha')
        ap1.addDV('mach',value=0.7,name='mach')

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.mp_group.s1.aero.set_ap(AP0)
        self.mp_group.s2.aero.set_ap(AP1)

        # add the structural thickness DVs
        self.dvs.add_output('dv_struct', np.array(self.as_group.n_dv_struct*[0.01]))
        self.mp_group.promote('s1', inputs=['dv_struct'])
        self.mp_group.promote('s2', inputs=['dv_struct'])
        self.connect('dv_struct', 'mp_group.dv_struct')

        # we can also add additional design variables, constraints and set the objective function here.
        # every solver is already initialized, so we can perform solver-specific calls
        # that are not in default MPHY API.

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile='as_configure_2scenario.html')
prob.run_model()
# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
    print('cl =',prob['mp_group.s0.aero.funcs.cl'])
    print('cd =',prob['mp_group.s0.aero.funcs.cd'])

    print("Scenario 1")
    print('cl =',prob['mp_group.s1.aero.funcs.cl'])
    print('cd =',prob['mp_group.s1.aero.funcs.cd'])
