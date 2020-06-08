#rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys.mphys_multipoint import MPHYS_Multipoint

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflow_builder
from mphys.mphys_tacs import TACS_builder
from mphys.mphys_meldthermal import MELDThermal_builder
from mphys.analysis_classes import CoupledAnalysis
# from mphys.mphys_rlt import RLT_builder

from baseclasses import *
from tacs import elements, constitutive, functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

# flag to use meld (False for RLT)
use_meld = True
# use_meld = False



# create builders 

################################################################################
# ADflow builder
################################################################################


adflow_options = {
    # 'printTiming': False,

    # Common Parameters
    'gridFile': '2dPlate_1e-02.cgns',
    'outputDirectory': './',
    # 'discretization': 'upwind',

    # 'oversetupdatemode': 'full',
    'volumevariables': ['temp'],
    'surfacevariables': ['cf', 'vx', 'vy', 'vz', 'temp', 'heattransfercoef', 'heatflux'],
    'monitorVariables':	['resturb', 'yplus', 'heatflux'],
    # Physics Parameters
    # 'equationType': 'laminar NS',
    'equationType': 'rans',
    # 'vis2':0.0,
    'liftIndex': 2,
    'CFL': 1.0,
    # 'smoother': 'DADI',
    # 'smoother': 'runge',

    'useANKSolver': True,
    'ANKswitchtol': 10e0,
    # 'ankcfllimit': 5e6,
    # 'anksecondordswitchtol': 5e-6,
    'ankcoupledswitchtol': 5e-6,
    # NK parameters
    'useNKSolver': False,
    'nkswitchtol': 1e-8,
    
    'rkreset': False,
    'nrkreset': 40,
    'MGCycle': 'sg',
    # 'MGStart': -1,
    # Convergence Parameters
    'L2Convergence': 1e-12,
    'nCycles': 1000,
    'nCyclesCoarse': 250,
    'ankcfllimit': 5e3,
    'nsubiterturb': 5,
    'ankphysicallstolturb': 0.99,
    'anknsubiterturb': 5,
    # 'ankuseturbdadi': False,
    'ankturbkspdebug': True,

    'storerindlayer': True,
    # Turbulence model
    'eddyvisinfratio': .210438,
    'useft2SA': False,
    'turbulenceproduction': 'vorticity',
    'useblockettes': False,

}

adflow_builder = ADflow_builder(adflow_options, heat_transfer=True )

################################################################################
# TACS builder
################################################################################
def add_elements(mesh):
    props = constitutive.MaterialProperties()
    con = constitutive.PlaneStressConstitutive(props)
    heat = elements.HeatConduction2D(con)

    # Create the basis class
    quad_basis = elements.LinearQuadBasis()

    # Create the element
    element = elements.Element2D(heat, quad_basis)

    mesh.setElement(0, element)

    ndof = heat.getVarsPerNode()


    return ndof, 0



def get_surface(tacs):

    # get structures nodes
    Xpts = tacs.createNodeVec()
    tacs.getNodes(Xpts)
    Xpts_array = Xpts.getArray()

    plate_surface = []
    mapping = []
    for i in range(len(Xpts_array) // 3):

        # check if it's on the flow edge
        if Xpts_array[3*i+1] == 0.0:
            plate_surface.extend(Xpts_array[3*i:3*i+3])
            mapping.append(i)


    plate_surface = np.array(plate_surface)
    return plate_surface, mapping

def get_funcs(tacs):
    ks_weight = 50.0
    return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

tacs_options = {
    'add_elements': add_elements,
    'mesh_file'   : 'flatplate.bdf',
    'get_funcs': get_funcs,
    'get_surface': get_surface,
}

tacs_builder = TACS_builder(tacs_options, conduction=True)

################################################################################
# Transfer scheme builder
################################################################################

if use_meld:
    xfer_options = {
        'isym': -1,
        'n': 10,
        'beta': 0.5,
    }

    xfer_builder = MELDThermal_builder(xfer_options, adflow_builder, tacs_builder)
else:
    # or we can use RLT:
    raise NotImplementedError
    # xfer_options = {'transfergaussorder': 2}
    # xfer_builder = RLT_builder(xfer_options, adflow_builder, tacs_builder)


# # define the multipoint analysis
# as_analysis = MPHYS_Multipoint(
#     aero_builder   = adflow_builder,
#     struct_builder = tacs_builder,
#     xfer_builder   = xfer_builder
# )




conjugate_hx = CoupledAnalysis(builders={
    'convection':adflow_builder,
    'conduction':tacs_builder,
    'xfer': xfer_builder

})

################################################################################
# Create OpenMDAO Model
################################################################################


class Top(om.Group):

    def setup(self):

        # ivc to keep the top level DVs
        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        dvs.add_output('alpha0', val=1.5)
        dvs.add_output('mach0', val=0.8)
        
   


        # create the multiphysics multipoint group.
        mp = self.add_subsystem(
            'conjugate', conjugate_hx

        )

        # self.connect('conv_mesh.x_a0', ['conjugate.x_a0'])
        # self.connect('cond_mesh.x_s0', ['conjugate.x_s0'])

        # this is the method that needs to be called for every point in this mp_group
        # mp.mphys_add_scenario('s0')

    def configure(self):
        # create the aero problems for both analysis point.
        # this is custom to the ADflow based approach we chose here.
        # any solver can have their own custom approach here, and we don't
        # need to use a common API. AND, if we wanted to define a common API,
        # it can easily be defined on the mp group, or the aero group.

        # define the aero DVs in the IVC
        # s0


        # ap0 = AeroProblem(
        #     name='ap0',
        #     mach=0.8,
        #     altitude=10000,
        #     alpha=1.5,
        #     areaRef=45.5,
        #     chordRef=3.25,
        #     evalFuncs=['lift','drag', 'cl', 'cd']
        # )
        # ap0.addDV('alpha',value=1.5,name='alpha')
        # ap0.addDV('mach',value=0.8,name='mach')
        # atmospheric conditions
        temp_air = 273  # kelvin
        Pr = 0.72
        mu = 1.81e-5  # kg/(m * s)

        u_inf = 68  # m/s\
        p_inf = 101e3


        ap = AeroProblem(name='fc_conv', V=u_inf, T=temp_air,
                        rho=1.225, areaRef=1.0, chordRef=1.0, alpha=0, beta=0,  evalFuncs=['cl', 'cd'])


        group = 'wall'
        BCVar = 'Temperature'

        CFDSolver = adflow_builder.get_solver()
        BCData = CFDSolver.getBCData(groupNames=[group])
        ap.setBCVar(BCVar,  BCData[group][BCVar], group)
        ap.addDV(BCVar, familyGroup=group, name='wall_temp')
        # import ipdb; ipdb.set_trace()

        # here we set the aero problems for every cruise case we have.
        # this can also be called set_flow_conditions, we don't need to create and pass an AP,
        # just flow conditions is probably a better general API
        # this call automatically adds the DVs for the respective scenario
        self.conjugate.convection.mphys_set_ap(ap)



        # connect to the aero for each scenario
        self.connect('conjugate.conduction_mesh.x_s0', ['conjugate.conduction.x_s0'])
        self.connect('conjugate.convection_mesh.x_a0', ['conjugate.convection.x_a0'])

        self.connect('conjugate.conduction_mesh.x_s0_surface' , ['conjugate.xfer_temps.x_s0',\
                                                                 'conjugate.xfer_heat_rate.x_s0'])

        self.connect('conjugate.convection_mesh.x_a0_surface', ['conjugate.xfer_temps.x_a0',\
                                                                'conjugate.xfer_heat_rate.x_a0'])

        self.connect('conjugate.xfer_temps.temp_conv', ['conjugate.convection.wall_temp_(1,1)'])
        self.connect('conjugate.conduction.temp_cond',['conjugate.xfer_temps.temp_cond'])

        self.connect('conjugate.convection.heatflux', ['conjugate.xfer_heat_rate.heat_xfer_conv'])
        self.connect('conjugate.xfer_heat_rate.heat_xfer_cond', ['conjugate.conduction.heat_xfer'])


        self.conjugate.nonlinear_solver = om.NonlinearBlockGS()
        self.conjugate.nonlinear_solver.options['maxiter'] = 2
        self.conjugate.nonlinear_solver.options['atol'] = 1e-8
        self.conjugate.nonlinear_solver.options['rtol'] = 1e-8
        self.conjugate.nonlinear_solver.options['iprint'] = 2
        # cycle.nonlinear_solver.options['use_aitken'] = True

        # self.connect('mach0', 'conjugate.mach')

        # add the structural thickness DVs
        # ndv_struct = self.conjugateruct_builder.get_ndv()
        # self.dvs.add_output('dv_struct', np.array(ndv_struct*[0.01]))
        # self.connect('dv_struct', ['conjugate.struct.dv_struct'])

        # we can also add additional design variables, constraints and set the objective function here.
        # every solver is already initialized, so we can perform solver-specific calls
        # that are not in default MPHYS API.

################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile='mphys_as.html')
prob.run_model()
# prob.model.list_outputs()
# if MPI.COMM_WORLD.rank == 0:
#     print("Scenario 0")
#     print('cl =',prob['mp_group.s0.aero.funcs.cl'])
#     print('cd =',prob['mp_group.s0.aero.funcs.cd'])
