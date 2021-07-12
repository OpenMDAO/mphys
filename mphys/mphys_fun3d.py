import openmdao.api as om

from mphys.builder import Builder
from mphys.mphys_group import MphysGroup
from libmeshdef.openmdao import MeshDeformationOpenMdao
from libmeshdef.mesh_deformer import MeshDeformer

from problem_setup_parser import ProblemSetupParser
from parfait.preprocessor import create_preprocessor
from parfait.distance_solver import DistanceSolver
from sfe.solver import SfeSolver
from sfe.om_component import SfeSolverOpenMdao, SfeForcesOpenMdao
from mphys.integrated_forces import IntegratedSurfaceForces
from pancake4py.iris import Iris


class Fun3dMesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare('meshdef_solver')
        self.options.declare('boundary_tag_list')

    def setup(self):
        boundary_tag_list = self.options['boundary_tag_list']
        meshdef = self.options['meshdef_solver']
        x_aero0 = meshdef.get_boundary_node_coordinates(boundary_tag_list,
                                                        owned_only=True,
                                                        concatenate=True)
        self.add_output('x_aero0', distributed=True, val=x_aero0.flatten(),
                        desc='initial aerodynamic surface node coordinates',
                        tags=['mphys_coordinates'])


class Fun3dFsiSolverGroup(MphysGroup):
    def initialize(self):
        self.options.declare('meshdef_comp')
        self.options.declare('flow_comp')
        self.options.declare('forces_comp')
        self.options.declare('number_of_surface_nodes')

    def setup(self):
        self.mphys_add_subsystem('meshdef', self.options['meshdef_comp'])
        self.mphys_add_subsystem('flow', self.options['flow_comp'])
        self.mphys_add_subsystem('forces',self.options['forces_comp'])


class Fun3dSfeBuilder(Builder):
    def __init__(self, boundary_tag_list, input_file='input.cfg'):
        self.boundary_tag_list = boundary_tag_list
        self.input_file = input_file

    def _initialize_meshdef(self, prob):
        return MeshDeformer(prob.problem, self.mesh, self.iris)

    def _initialize_sfe(self, prob):
        return SfeSolver(prob.problem, self.mesh, self.iris)

    def initialize(self, comm):
        self.iris = Iris(comm)
        prob = ProblemSetupParser(self.iris, self.input_file)
        prob.read_bc_tags_from_mapbc()

        self.mesh = create_preprocessor(prob.problem, self.iris)
        self.sfe = self._initialize_sfe(prob)
        self.meshdef = self._initialize_meshdef(prob)
        self.number_of_nodes = self.meshdef.get_boundary_node_global_ids(
            self.boundary_tag_list, owned_only=True).size

        self._set_wall_distance(prob)

    def _set_wall_distance(self, prob):
        dist_solver = DistanceSolver(prob.problem, self.mesh, self.iris)
        distance = dist_solver.get_wall_distance()
        self.sfe.set_node_wall_distance(distance, owned_only=False)
        self.meshdef.set_node_wall_distance(distance, owned_only=False)

    def get_mesh_coordinate_subsystem(self):
        return Fun3dMesh(meshdef_solver=self.meshdef,
                         boundary_tag_list=self.boundary_tag_list)

    def get_coupling_group_subsystem(self):
        meshdef_om = MeshDeformationOpenMdao(meshdef_solver=self.meshdef,
                                             boundary_tag_list=self.boundary_tag_list)
        sfe_om = SfeSolverOpenMdao(sfe_solver=self.sfe)
        forces_om = SfeForcesOpenMdao(sfe_solver=self.sfe,
                                      boundary_tag_list=self.boundary_tag_list)
        return Fun3dFsiSolverGroup(meshdef_comp=meshdef_om,
                                   flow_comp=sfe_om,
                                   forces_comp=forces_om,
                                   number_of_surface_nodes=self.number_of_nodes)

    def get_post_coupling_subsystem(self):
        return IntegratedSurfaceForces()

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def get_ndof(self):
        return 3
