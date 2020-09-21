#rst Imports
from __future__ import print_function, division
import os

import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys.multipoint import Multipoint

from parameterized import parameterized, parameterized_class
import unittest

from openmdao.utils.assert_utils import assert_near_equal


# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflowBuilder
from mphys.mphys_tacs import TacsBuilder
from mphys.mphys_meld import MeldBuilder
from mphys.mphys_rlt import RltBuilder

from baseclasses import *
from tacs import elements, constitutive, functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

# flag to use meld (False for RLT)
# use_meld = True
use_meld = False


baseDir = os.path.dirname(os.path.abspath(__file__))


class Top(om.Group):

    def setup(self):

        ################################################################################
        # ADflow options
        ################################################################################
        aero_options = {
            # I/O Parameters

            'gridFile':os.path.join(baseDir, '../input_files/wing_vol.cgns'),
            'outputDirectory':'.',
            'monitorvariables':['resrho','resturb','cl','cd'],
            'writeTecplotSurfaceSolution':False,
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
            'nsubiterturb': 5,
            'anksecondordswitchtol':1e-4,
            'ankcoupledswitchtol': 1e-6,
            'ankinnerpreconits':2,
            'ankouterpreconits':2,
            'anklinresmax': 0.1,

            # Termination Criteria
            'L2Convergence':1e-14,
            'L2ConvergenceCoarse':1e-2,
            'L2ConvergenceRel': 1e-4,
            'nCycles':10000,

            # force integration
            'forcesAsTractions': self.forcesAsTractions,

        }




        adflow_builder = ADflowBuilder(aero_options)

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
            'mesh_file'   : os.path.join(baseDir, '../input_files/wingbox.bdf'),
            'get_funcs'   : get_funcs
        }

        tacs_builder = TacsBuilder(tacs_options)

        ################################################################################
        # Transfer scheme options
        ################################################################################
        xfer_builder = self.xfer_builder_class( self.xfer_options, adflow_builder, tacs_builder,check_partials=True )

        ################################################################################
        # MPHYS setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])

        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'mp_group',
            Multipoint(
                aero_builder   = adflow_builder,
                struct_builder = tacs_builder,
                xfer_builder   = xfer_builder
            )
        )

        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario('s0')

    def configure(self):
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

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.mp_group.s0.solver_group.aero.mphys_set_ap(ap0)
        self.mp_group.s0.aero_funcs.mphys_set_ap(ap0)

        # define the aero DVs in the IVC
        # s0
        self.dvs.add_output('alpha0', val=1.5)
        self.dvs.add_output('mach0', val=0.8)

        # connect to the aero for each scenario
        self.connect('alpha0', ['mp_group.s0.solver_group.aero.alpha', 'mp_group.s0.aero_funcs.alpha'])
        self.connect('mach0', ['mp_group.s0.solver_group.aero.mach', 'mp_group.s0.aero_funcs.mach'])

        # add the structural thickness DVs
        ndv_struct = self.mp_group.struct_builder.get_ndv()
        self.dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))
        self.connect('dv_struct', ['mp_group.s0.solver_group.struct.dv_struct', 'mp_group.s0.struct_funcs.dv_struct',])

        # we can also add additional design variables, constraints and set the objective function here.
        # every solver is already initialized, so we can perform solver-specific calls
        # that are not in default MPHYS API.


@parameterized_class([
    {
        "name":'meld',
        "xfer_builder_class": MeldBuilder,
        "xfer_options":{'isym': 1,
                        'n': 200,
                        'beta': 0.5},
        "ref_vals":{'xa' : 5.443441765975671,
                      'cl' :  0.33844664,
                      'f_struct': 0.24355979,
                      'cd' :  0.02988495,
        }
    },
    {
        "name":'rlt',
        "xfer_builder_class": RltBuilder,
        "xfer_options": {'transfergaussorder': 2},
        "ref_vals":{'xa' : 5.504999831790868,
                    'f_struct': 0.31363742,
                      'cl' :  0.3047756,
                      'cd' :  0.0280476
        }
    }
    ])
class TestAeroStructSolve(unittest.TestCase):
    N_PROCS = 1
    def setUp(self):

        ################################################################################
        # OpenMDAO setup
        ################################################################################

        prob = om.Problem()
        prob.model = Top()

        prob.model.xfer_builder_class = self.xfer_builder_class
        prob.model.xfer_options = self.xfer_options

        if "meld" in self.name:
            prob.model.forcesAsTractions = True
        else:
            prob.model.forcesAsTractions = False


        model = prob.model
        prob.setup()

        self.prob = prob
        om.n2(prob, show_browser=False, outfile='test_as.html')

    def test_run_model(self):
        self.prob.run_model()
        # prob.model.list_outputs()
        if MPI.COMM_WORLD.rank == 0:
            print("Scenario 0")

            print('xa =', np.mean(self.prob.get_val('mp_group.s0.solver_group.aero.geo_disp.x_a', get_remote=True)))
            print('cl =',   self.prob.get_val('mp_group.s0.aero_funcs.cl', get_remote=True))
            print('cd =',   self.prob.get_val('mp_group.s0.aero_funcs.cd', get_remote=True))
            print('cd =',   self.prob.get_val('mp_group.s0.struct_funcs.funcs.f_struct', get_remote=True))

            assert_near_equal(np.mean(self.prob.get_val('mp_group.s0.solver_group.aero.geo_disp.x_a', get_remote=True)), self.ref_vals['xa'], 1e-6)
            assert_near_equal(np.mean(self.prob.get_val('mp_group.s0.aero_funcs.cl', get_remote=True)), self.ref_vals['cl'], 1e-6)
            assert_near_equal(np.mean(self.prob.get_val('mp_group.s0.aero_funcs.cd', get_remote=True)), self.ref_vals['cd'], 1e-6)
            assert_near_equal(np.mean(self.prob.get_val('mp_group.s0.struct_funcs.funcs.f_struct', get_remote=True)), self.ref_vals['f_struct'], 1e-6)

