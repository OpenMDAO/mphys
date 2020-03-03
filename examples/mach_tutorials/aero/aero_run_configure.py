import numpy as np

import openmdao.api as om

#from om_mach import AeroSolver
from adflow import OM_ADFLOW as ADFLOW_OM

class AeroSolver(om.Group):

    def initialize(self):
        # we need an aero solver
        self.options.declare('aero', allow_none=False)
        self.options.declare('aeroOpts', allow_none=False)
        # we may have a component for mesh warping embedded in the aero group
        self.options.declare('meshOpts', allow_none=True, default=None)

    def setup(self):

        # check if we have a geometry component
        if self.options['geo']:
            geoClass = self.options['geo']
            geoOpts = self.options['geoOpts']
            self.add_subsystem('geo', geoClass(geoOpts=geoOpts))

        # we always have an aerodynamic solver
        aeroClass = self.options['aero']
        aeroOpts = self.options['aeroOpts']
        meshOpts = self.options['meshOpts']
        self.add_subsystem('aero', aeroClass(aeroOpts=aeroOpts, meshOpts=meshOpts))



class Top(om.Group):
    
    def setup(self):
        
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
            'MGCycle':'3w',
            'MGStartLevel':-1,
            'nCyclesCoarse':250,

            # ANK Solver Parameters
            'useANKSolver':True,
            'ankswitchtol':1e-1,

            # NK Solver Parameters
            'useNKSolver':True,
            'nkswitchtol':1e-4,

            # Termination Criteria
            'L2Convergence':1e-12,
            'L2ConvergenceCoarse':1e-2,
            'nCycles':1000,
        }

        self.add_subsystem('aero_adflow', AeroSolver(aero=ADFLOW_OM, aeroOpts=aero_options))


    def configure(self):
        aero = self.aero_adflow.aero

        # this methods is adflow specific 
        aero.setAeroProblem(name='wing', mach=0.8, altitude=10000, alpha=1.5,
                            areaRef=45.5, chordRef=3.25, evaluFuncs=['cl','cd'])
        
# OpenMDAO set up
prob = om.Problem()

prob.model = Top()

# model.nonlinear_solver = om.NonlinearRunOnce()
# model.linear_solver = om.LinearRunOnce()

prob.setup()
prob.run_model()

if prob.model.comm.rank == 0:
    print('cl =',prob['aero_adflow.aero.functionals.cl'])
    print('cd =',prob['aero_adflow.aero.functionals.cd'])


