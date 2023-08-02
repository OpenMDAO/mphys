import openmdao.api as om
from .scenario import Scenario
from .coupling_aerothermal import CouplingAeroThermal


class ScenarioAeroThermal(Scenario):

    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        super().initialize()
        
        self.options.declare('aero_builder', recordable=False,
                              desc='The Mphys builder for the aerodynamic solver')
        self.options.declare('thermal_builder', recordable=False,
                              desc='The Mphys builder for the Thermal solver')
        self.options.declare('thermalxfer_builder', recordable=False,
                              desc='The Mphys builder for the heat flux and temperature transfer')
        self.options.declare('in_MultipointParallel', default=False,
                             desc='Set to `True` if adding this scenario inside a MultipointParallel Group.')
        self.options.declare('geometry_builder', default=None, recordable=False,
                             desc='The optional Mphys builder for the geometry')

    def _mphys_scenario_setup(self):
        aero_builder = self.options['aero_builder']
        thermal_builder = self.options['thermal_builder']
        thermalxfer_builder = self.options['thermalxfer_builder']
        geometry_builder = self.options['geometry_builder']

        if self.options['in_MultipointParallel']:
            self._mphys_initialize_builders(aero_builder, thermal_builder,
                                            thermalxfer_builder, geometry_builder)
            self._mphys_add_mesh_and_geometry_subsystems(aero_builder, thermal_builder,
                                                         geometry_builder)

        self._mphys_add_pre_coupling_subsystem_from_builder('aero', aero_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder('thermal', thermal_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder('thermalxfer', thermalxfer_builder, self.name)

        coupling_group = CouplingAeroThermal(aero_builder=aero_builder,
                                                thermal_builder=thermal_builder,
                                                thermalxfer_builder=thermalxfer_builder,
                                                scenario_name=self.name)
        self.mphys_add_subsystem('coupling', coupling_group)

        self._mphys_add_post_coupling_subsystem_from_builder('thermalxfer', thermalxfer_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder('aero', aero_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder('thermal', thermal_builder, self.name)

    def _mphys_initialize_builders(self, aero_builder, thermal_builder,
                                   thermalxfer_builder, geometry_builder):
        aero_builder.initialize(self.comm)
        thermal_builder.initialize(self.comm)
        thermalxfer_builder.initialize(self.comm)
        if geometry_builder is not None:
            geometry_builder.initialize(self.comm)

    def _mphys_add_mesh_and_geometry_subsystems(self, aero_builder, thermal_builder,
                                                geometry_builder):

        if geometry_builder is None:
            self.mphys_add_subsystem('aero_mesh', aero_builder.get_mesh_coordinate_subsystem(self.name))
            self.mphys_add_subsystem('thermal_mesh', thermal_builder.get_mesh_coordinate_subsystem(self.name))
        else:
            self.add_subsystem('aero_mesh', aero_builder.get_mesh_coordinate_subsystem(self.name))
            self.add_subsystem('thermal_mesh', thermal_builder.get_mesh_coordinate_subsystem(self.name))
            self.mphys_add_subsystem('geometry', geometry_builder.get_mesh_coordinate_subsystem(self.name))
            self.connect('aero_mesh.x_aero0', 'geometry.x_aero_in')
            self.connect('thermal_mesh.x_thermal0', 'geometry.x_thermal_in')
