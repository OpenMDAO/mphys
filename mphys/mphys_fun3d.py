import numpy as np
import openmdao.api as om

from mphys.base_classes import SolverBuilder

from iris_wrapper import Iris
from parfait.distance_calculator import DistanceCalculator
from libmeshdef.meshdef_parfait import MeshDeformation
from libmeshdef.meshdef_openmdao import MeshDeformationOpenMdao
from sfe.sfe_parfait import SFE
from sfe.sfe_openmdao import SfeSolverOpenMdao, SfeForcesOpenMdao
from mphys.integrated_forces import IntegratedSurfaceForces
from mphys.geo_disp import GeoDisp

class Fun3dMesh(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('meshdef_solver')
        self.options.declare('boundary_tag_list')
        self.options['distributed'] = True

    def setup(self):
        boundary_tag_list = self.options['boundary_tag_list']
        meshdef = self.options['meshdef_solver']
        x,y,z = meshdef.get_boundary_node_coordinates(boundary_tag_list, owned_only = True)
        self.x_a0 = self._flatten_vectors(x,y,z)
        coord_size = self.x_a0.size
        self.add_output('x_a0', shape=coord_size, desc='initial aerodynamic surface node coordinates')

    def _flatten_vectors(self, x, y, z):
        matrix = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1)),z.reshape((-1,1))),axis=1)
        return matrix.flatten(order="C")

    def mphys_add_coordinate_input(self):
        local_size = self.x_a0.size
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a0_points',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')

        return 'x_a0_points', self.x_a0

    def compute(self,inputs,outputs):
        if 'x_a0_points' in inputs:
            outputs['x_a0'] = inputs['x_a0_points']
        else:
            outputs['x_a0'] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'x_a0_points' in d_inputs:
                d_outputs['x_a0'] += d_inputs['x_a0_points']
        elif mode == 'rev':
            if 'x_a0_points' in d_inputs:
                d_inputs['x_a0_points'] += d_outputs['x_a0']

class Fun3dFsiSolverGroup(om.Group):
    def initialize(self):
        self.options.declare('meshdef_comp')
        self.options.declare('flow_comp')
        self.options.declare('forces_comp')
        self.options.declare('number_of_surface_nodes')

    def setup(self):
        geodisp_comp = GeoDisp(number_of_surface_nodes=self.options['number_of_surface_nodes'])
        meshdef_comp = self.options['meshdef_comp']
        flow_comp = self.options['flow_comp']
        forces_comp = self.options['forces_comp']

        self.add_subsystem('geo_disp', geodisp_comp,
                                       promotes_inputs=['u_a', 'x_a0'],
                                       promotes_outputs=['x_a'])
        self.add_subsystem('meshdef', meshdef_comp)
        self.add_subsystem('flow', flow_comp)
        self.add_subsystem('forces', forces_comp,
                                     promotes_outputs=['f_a'])

    def configure(self):
        self.connect('x_a','meshdef.x_a')
        self.connect('meshdef.u_g',['flow.u_g','forces.u_g'])
        self.connect('flow.q','forces.q')

class Fun3dSfeBuilder(SolverBuilder):
    def __init__(self, meshfile, boundary_tag_list):
        self.meshfile = meshfile
        self.boundary_tag_list = boundary_tag_list

    def init_solver(self, comm):
        self.dist_calc = DistanceCalculator.from_meshfile(self.meshfile,comm)
        distance = self.dist_calc.compute(self.boundary_tag_list)

        self.iris = Iris(comm)
        self.sfe = SFE(self.dist_calc.mesh,self.iris)
        self.meshdef = MeshDeformation(self.dist_calc.mesh,self.iris)
        self.nnodes = self.meshdef.get_boundary_node_global_ids(self.boundary_tag_list, owned_only=True).size

        self.sfe.set_node_wall_distance(distance, owned_only = False)
        self.meshdef.set_node_wall_distance(distance, owned_only = False)

    def get_mesh_element(self):
        return Fun3dMesh(meshdef_solver = self.meshdef,
                         boundary_tag_list = self.boundary_tag_list)

    def get_element(self, **kwargs):
        meshdef_om = MeshDeformationOpenMdao(meshdef_solver = self.meshdef,
                                             boundary_tag_list = self.boundary_tag_list)
        sfe_om = SfeSolverOpenMdao(sfe_solver = self.sfe)
        forces_om = SfeForcesOpenMdao(sfe_solver = self.sfe,
                                      boundary_tag_list = self.boundary_tag_list)

        return Fun3dFsiSolverGroup(meshdef_comp = meshdef_om,
                                   flow_comp = sfe_om,
                                   forces_comp = forces_om,
                                   number_of_surface_nodes = self.nnodes)
    def get_scenario_element(self):
        return IntegratedSurfaceForces(number_of_surface_nodes=self.nnodes)

    def get_nnodes(self):
        return self.nnodes

    def get_scenario_connections(self):
        mydict = {
            'f_a': 'f_a',
            'x_a': 'x_a'
        }
        return mydict

    def get_mesh_connections(self):
        mydict = {
            'solver':{
                'x_a0'  : 'x_a0',
            },
            'funcs':{},
        }
        return mydict
