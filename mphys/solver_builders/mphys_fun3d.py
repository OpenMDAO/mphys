import openmdao.api as om

from mphys.builder import Builder
from libmeshdef.openmdao import MeshDeformationOpenMdao
from libmeshdef.mesh_deformer import MeshDeformer

from problem_setup_parser import ProblemSetupParser
from parfait.preprocessor import create_preprocessor
from parfait.distance_solver import DistanceSolver
from sfe.om_component import SfeSolverOpenMdao, SfeForcesOpenMdao
from sfe.solver import SfeSolver
from mphys.integrated_forces import IntegratedSurfaceForces
from pancake4py.iris import Iris

class Fun3dMesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare('meshdef_solver')
        self.options.declare('boundary_tag_list')

    def setup(self):
        boundary_tag_list = self.options['boundary_tag_list']
        meshdef = self.options['meshdef_solver']
        x_aero0 = meshdef.get_boundary_node_coordinates(boundary_tag_list, owned_only = True, concatenate=True)
        self.add_output('x_aero0', distributed=True, val=x_aero0.flatten(), desc='initial aerodynamic surface node coordinates',
                         tags=['mphys_coordinates'])

class Fun3dFsiSolverGroup(om.Group):
    def initialize(self):
        self.options.declare('meshdef_comp')
        self.options.declare('flow_comp')
        self.options.declare('forces_comp')
        self.options.declare('number_of_surface_nodes')

    def setup(self):
        meshdef_comp = self.options['meshdef_comp']
        flow_comp = self.options['flow_comp']
        forces_comp = self.options['forces_comp']

        self.add_subsystem('meshdef', meshdef_comp, promotes_inputs=['x_aero'])
        self.add_subsystem('flow', flow_comp, promotes_inputs=['mach','reynolds','aoa','yaw'])
        self.add_subsystem('forces', forces_comp,
                                     promotes_inputs=['mach','reynolds','q_inf'],
                                     promotes_outputs=['f_aero'])

    def configure(self):
        self.connect('meshdef.u_g',['flow.u_g','forces.u_g'])
        self.connect('flow.q','forces.q')

class Fun3dSfeBuilder(Builder):
    def __init__(self, boundary_tag_list, input_file='input.cfg'):
        self.boundary_tag_list = boundary_tag_list
        self.input_file = input_file

    def initialize(self, comm):
        iris = Iris(comm)
        prob = ProblemSetupParser(iris, self.input_file)

        self.mesh = create_preprocessor(prob.problem, iris)
        self.sfe = SfeSolver(prob.problem, self.mesh, iris)
        self.meshdef = MeshDeformer(prob.problem, self.mesh, iris)
        self.number_of_nodes = self.meshdef.get_boundary_node_global_ids(self.boundary_tag_list, owned_only=True).size

        dist_solver = DistanceSolver(prob.problem, self.mesh, iris)
        distance = dist_solver.get_wall_distance()
        self.sfe.set_node_wall_distance(distance, owned_only = False)
        self.meshdef.set_node_wall_distance(distance, owned_only = False)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return Fun3dMesh(meshdef_solver = self.meshdef,
                         boundary_tag_list = self.boundary_tag_list)

    def get_coupling_group_subsystem(self, scenario_name=None):
        meshdef_om = MeshDeformationOpenMdao(meshdef_solver = self.meshdef,
                                             boundary_tag_list = self.boundary_tag_list)
        sfe_om = SfeSolverOpenMdao(sfe_solver = self.sfe)
        forces_om = SfeForcesOpenMdao(sfe_solver = self.sfe,
                                      boundary_tag_list = self.boundary_tag_list)
        return Fun3dFsiSolverGroup(meshdef_comp = meshdef_om,
                                   flow_comp = sfe_om,
                                   forces_comp = forces_om,
                                   number_of_surface_nodes = self.number_of_nodes)
    def get_post_coupling_subsystem(self, scenario_name=None):
        return IntegratedSurfaceForces()

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def get_ndof(self):
        return 3
