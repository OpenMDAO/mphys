import openmdao.api as om
import numpy as np

from libmeshdef.openmdao import MeshDeformationOpenMdao

from mphys.mphys_group import MphysGroup
from .mphys_fun3d import Fun3dSfeBuilder, Fun3dFsiSolverGroup
from sfe.om_component import SfeSolverOpenMdao, SfeForcesOpenMdao
from sfe.lfd_solver import SfeLfdSolver
from sfe.lfd_om_component import LfdOpenMdao
from pk_flutter_solver.om_component import PkSolverOM
from libmeshdef.openmdao import MeshDeformer
from parfait.distance_solver import DistanceSolver


class PkFlutterDescription:
    def __init__(self, boundary_tag_list, nmodes, reduced_frequencies, u_ref, semichord,
                 pk_density, pk_velocity, modal_reference_amplitudes=0.01):
        self.boundary_tag_list = boundary_tag_list
        self.nmodes = nmodes
        self.reduced_frequencies = reduced_frequencies
        self.u_ref = u_ref
        self.semichord = semichord
        self.pk_density = pk_density
        self.pk_velocity = pk_velocity
        self.modal_reference_amplitude = modal_reference_amplitudes


class Fun3dLfdFlutterGroup(MphysGroup):
    def initialize(self):
        self.options.declare('sfe_solver')
        self.options.declare('meshdef_solver')
        self.options.declare('iris')
        self.options.declare('problem_description')

    def setup(self):
        prob = self.options['problem_description']
        sfe = self.options['sfe_solver']
        meshdef = self.options['meshdef_solver']
        iris = self.options['iris']

        lfd_frequencies = prob.reduced_frequencies * prob.u_ref / prob.semichord
        self.mphys_add_subsystem('lfd',
                                 LfdOpenMdao(sfe_solver=sfe,
                                             meshdef_solver = meshdef,
                                             iris=iris,
                                             boundary_tag_list=prob.boundary_tag_list,
                                             nmodes=prob.nmodes,
                                             lfd_frequencies=lfd_frequencies,
                                             modal_ref_amplitude=prob.modal_reference_amplitude,
                                             u_ref=prob.u_ref))
        self.mphys_add_subsystem('pk',
                                 PkSolverOM(density=prob.pk_density,
                                            velocity=prob.pk_velocity,
                                            reduced_frequencies=prob.reduced_frequencies,
                                            nmodes=prob.nmodes,
                                            b_ref=prob.semichord,
                                            u_ref=prob.u_ref))


class Fun3dLfdBuilder(Fun3dSfeBuilder):
    def __init__(self, boundary_tag_list, nmodes, u_ref, semichord,
                 reduced_frequencies, pk_density, pk_velocity,
                 modal_reference_amplitude=0.01,
                 input_file='input.cfg'):
        self.nmodes = nmodes
        self.u_ref = u_ref
        self.semichord = semichord
        self.reduced_frequencies = reduced_frequencies
        self.pk_density = pk_density
        self.pk_velocity = pk_velocity
        self.modal_reference_amplitude = modal_reference_amplitude

        super().__init__(boundary_tag_list, input_file)

    def _initialize_meshdef(self, prob):
        return MeshDeformer(prob.problem, self.mesh, self.iris, library_basename='libmeshdef_wrapper_complex')

    def _initialize_sfe(self, prob):
        return SfeLfdSolver(prob.problem, self.mesh, self.iris)

    def _set_wall_distance(self, prob):
        dist_solver = DistanceSolver(prob.problem, self.mesh, self.iris)
        distance = np.complex128(dist_solver.get_wall_distance())
        self.sfe.set_node_wall_distance(distance, owned_only=False)
        self.meshdef.set_node_wall_distance(distance, owned_only=False)

    def initialize(self, comm):
        super().initialize(comm)
        self.prob = PkFlutterDescription(self.boundary_tag_list, self.nmodes,
                                         self.reduced_frequencies, self.u_ref,
                                         self.semichord,
                                         self.pk_density, self.pk_velocity,
                                         self.modal_reference_amplitude)

    def get_coupling_group_subsystem(self):
        meshdef_om = MeshDeformationOpenMdao(meshdef_solver=self.meshdef,
                                             boundary_tag_list=self.boundary_tag_list,
                                             complex_mode=True)
        sfe_om = SfeSolverOpenMdao(sfe_solver=self.sfe, complex_mode=True)
        forces_om = SfeForcesOpenMdao(sfe_solver=self.sfe,
                                       boundary_tag_list=self.boundary_tag_list,
                                       complex_mode = True)
        return Fun3dFsiSolverGroup(meshdef_comp=meshdef_om,
                                   flow_comp=sfe_om,
                                   forces_comp=forces_om,
                                   number_of_surface_nodes=self.number_of_nodes)

    def get_post_coupling_subsystem(self):
        return Fun3dLfdFlutterGroup(sfe_solver = self.sfe, meshdef_solver = self.meshdef,
                                    iris=self.iris, problem_description = self.prob)
