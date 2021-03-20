import openmdao.api as om
from .scenario import Scenario
from .coupling_aerostructural import CouplingAeroStructural


class ScenarioAeroStructural(Scenario):

    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        self.options.declare('aero_builder', recordable=False,
                              desc='The Mphys builder for the aerodynamic solver')
        self.options.declare('struct_builder', recordable=False,
                              desc='The Mphys builder for the structural solver')
        self.options.declare('ldxfer_builder', recordable=False,
                              desc='The Mphys builder for the load and displacement transfer')
        self.options.declare('in_MultipointParallel', default=False,
                             desc='Set to `True` if adding this scenario inside a MultipointParallel Group.')
        self.options.declare('geometry_builder', default=None, recordable=False,
                             desc='The optional Mphys builder for the geometry')

    def setup(self):
        aero_builder = self.options['aero_builder']
        struct_builder = self.options['struct_builder']
        ldxfer_builder = self.options['ldxfer_builder']
        geometry_builder = self.options['geometry_builder']

        if self.options['in_MultipointParallel']:
            self._mphys_initialize_builders(aero_builder, struct_builder,
                                            ldxfer_builder, geometry_builder)
            self._mphys_add_mesh_and_geometry_subsystems(aero_builder, struct_builder,
                                                         geometry_builder)

        self.mphys_add_pre_coupling_subsystem('aero', aero_builder)
        self.mphys_add_pre_coupling_subsystem('struct', struct_builder)
        self.mphys_add_pre_coupling_subsystem('ldxfer', ldxfer_builder)

        coupling_group = CouplingAeroStructural(aero_builder=aero_builder,
                                                struct_builder=struct_builder,
                                                ldxfer_builder=ldxfer_builder)
        self.mphys_add_subsystem('coupling', coupling_group)

        self.mphys_add_post_coupling_subsystem('aero', aero_builder)
        self.mphys_add_post_coupling_subsystem('struct', struct_builder)
        self.mphys_add_post_coupling_subsystem('ldxfer', ldxfer_builder)

    def _mphys_initialize_builders(self, aero_builder, struct_builder,
                                   ldxfer_builder, geometry_builder):
        aero_builder.initialize(self.comm)
        struct_builder.initialize(self.comm)
        ldxfer_builder.initialize(self.comm)
        if geometry_builder is not None:
            geometry_builder.initialize(self.comm)

    def _mphys_add_mesh_and_geometry_subsystems(self, aero_builder, struct_builder,
                                                geometry_builder):
        self.mphys_add_subsystem('mesh_aero', aero_builder.get_mesh_coordinate_subsystem())
        self.mphys_add_subsystem('mesh_struct', struct_builder.get_mesh_coordinate_subsystem())

        if geometry_builder is not None:
            self.mphys_add_subsystem('geometry', geometry_builder.get_mesh_coordinate_subsystem())
            self.connect('mesh_aero.x_aero0', 'geometry.x_aero_in')
            self.connect('mesh_struct.x_struct0', 'geometry.x_struct_in')
