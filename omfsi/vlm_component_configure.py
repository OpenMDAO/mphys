from __future__ import division, print_function
import numpy as np

import openmdao.api as om
from openmdao.api import ImplicitComponent, ExplicitComponent, Group
from omfsi.geo_disp import Geo_Disp

# TODO uncomment this to get the actual VLM_solver and forces comps
# from vlm_solver import VLM_solver, VLM_forces


class dummyVLMSolver(object):
    """ A dummy class that imitates a VLM solver object """

    def __init__(self, options, comm):

        # setup the VLM solver with a comm and options
        self.comm = comm
        self.options=options

        # do stuff
        self.n_nodes = self.options['N_nodes']

    def get_n_nodes(self):
        return self.n_nodes

    def getSurfaceCoordinates(self):
        return np.zeros(self.n_nodes * 3)


class VlmMesh(ExplicitComponent):

    def initialize(self):

        self.options.declare('aero_solver')

    def setup(self):

        self.aero_solver = self.options['aero_solver']
        N_nodes = self.aero_solver.get_n_nodes()
        self.x_a0 = self.aero_solver.getSurfaceCoordinates()
        self.add_output('x_a0_mesh',np.zeros(N_nodes*3))

    def compute(self,inputs,outputs):

        outputs['x_a0_mesh'] = self.x_a0

# TODO remove this class once the actual VLM code is present
class VLM_solver(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('aero_solver')

    def setup(self):
        self.aero_solver = self.options['aero_solver']
        n_nodes = self.aero_solver.get_n_nodes()
        self.add_output('q', shape=n_nodes*3)
        self.add_input('x_g', shape=n_nodes*3)

    def mphy_add_aero_dv(self, name):
        self.add_input(name)


# TODO remove this class once the actual VLM code is present
class VLM_forces(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('aero_solver')

    def setup(self):
        self.aero_solver = self.options['aero_solver']
        N_nodes = self.aero_solver.get_n_nodes()
        self.f_a = self.aero_solver.getSurfaceCoordinates()
        self.add_input('q', shape=N_nodes*3)
        self.add_input('x_g', shape=N_nodes*3)
        self.add_output('f_a',np.zeros(N_nodes*3))


    def mphy_add_aero_dv(self, name):
        self.add_input(name)

class VlmOutput(ExplicitComponent):

    def initialize(self):
        self.options.declare('aero_solver')

    def setup(self):
        self.aero_solver = self.options['aero_solver']
        n_nodes = self.aero_solver.get_n_nodes()
        self.add_input('q', shape=n_nodes*3)
        self.add_input('x_g', shape=n_nodes*3)

        self.add_input('CL',0.0)
        self.add_input('CD',0.0)
        self.add_output('CL_out',0.0)
        self.add_output('CD_out',0.0)
        self.declare_partials('CL_out','CL')
        self.declare_partials('CL_out','CD')
        self.declare_partials('CD_out','CL')
        self.declare_partials('CD_out','CD')

    def compute(self,inputs,outputs):

        outputs['CL_out'] = inputs['CL']
        outputs['CD_out'] = inputs['CD']

    def compute_partials(self,inputs,partials):

        partials['CL_out','CL'] = 1.0
        partials['CL_out','CD'] = 0.0
        partials['CD_out','CL'] = 0.0
        partials['CD_out','CD'] = 1.0

    def mphy_add_aero_dv(self, name):
        self.add_input(name)

class VLM_group(Group):

    def initialize(self):
        self.options.declare('solver')

    def setup(self):

        self.aero_solver = self.options['solver']

        self.add_subsystem('geo_disp', Geo_Disp(
            nnodes=int(self.aero_solver.getSurfaceCoordinates().size /3)),
            promotes_inputs=['u_a', 'x_a0']
        )
        self.add_subsystem('solver', VLM_solver(
            aero_solver=self.aero_solver
        ))
        self.add_subsystem('force', VLM_forces(
            aero_solver=self.aero_solver),
            promotes_outputs=['f_a']
        )
        self.add_subsystem('funcs', VlmOutput(
            aero_solver=self.aero_solver
        ))

    def configure(self):
        self.connect('geo_disp.x_a', ['solver.x_g', 'force.x_g', 'funcs.x_g'])
        self.connect('solver.q', ['force.q', 'funcs.q'])

    def mphy_set_ap(self, ap):
        # set the ap, add inputs and outputs, promote?
        self.solver.set_ap(ap)
        self.force.set_ap(ap)
        self.funcs.set_ap(ap)

        # promote the DVs for this ap
        ap_vars,_ = get_dvs_and_cons(ap=ap)

        for (args, kwargs) in ap_vars:
            name = args[0]
            size = args[1]
            self.promotes('solver', inputs=[name])
            self.promotes('force', inputs=[name])
            self.promotes('funcs', inputs=[name])

    def mphy_add_aero_dv(self, name):
        # set the ap, add inputs and outputs, promote?
        self.solver.mphy_add_aero_dv(name)
        self.force.mphy_add_aero_dv(name)
        self.funcs.mphy_add_aero_dv(name)

        # promote the DVs for this ap
        self.promotes('solver', inputs=[name])
        self.promotes('force', inputs=[name])
        self.promotes('funcs', inputs=[name])

class VLM_builder(object):

    def __init__(self, options):
        self.options = options


        # self.comm = comm

        self.mach = options['mach']
        self.q_inf = options['q_inf']
        self.vel = options['vel']
        self.mu = options['mu']

        ## mesh parameters

        self.N_nodes = options['N_nodes']
        self.N_elements = options['N_elements']
        self.x_a0 = options['x_a0']
        self.quad = options['quad']

    # api level method for all builders
    def init_solver(self, comm):
        self.solver = dummyVLMSolver(options=self.options, comm=comm)

    # api level method for all builders
    def get_solver(self):
        return self.solver

    # api level method for all builders
    def get_element(self):
        return VLM_group(solver=self.solver)

    def get_mesh_element(self):
        return VlmMesh(aero_solver=self.solver)

    def get_nnodes(self):
        return int(self.solver.getSurfaceCoordinates().size /3)