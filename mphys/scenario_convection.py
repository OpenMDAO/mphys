from .scenario import Scenario

class ScenarioAerodynamic(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        self.options.declare('aero_builder', recordable=False,
                             desc='The Mphys builder for the aerodynamic solver')
        self.options.declare('in_MultipointParallel', default=False, types=bool,
                             desc='Set to `True` if adding this scenario inside a MultipointParallel Group.')
        self.options.declare('geometry_builder', default=None, recordable=False,
                             desc='The optional Mphys builder for the geometry')

    def setup(self):
        aero_builder = self.options['aero_builder']
        geometry_builder = self.options['geometry_builder']

        if self.options['in_MultipointParallel']:
            aero_builder.initialize(self.comm)

            if geometry_builder is not None:
                geometry_builder.initialize(self.comm)
                self.add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem(self.name))
                self.mphys_add_subsystem('geometry',geometry_builder.get_mesh_coordinate_subsystem(self.name))
                self.connect('mesh.x_aero0','geometry.x_aero_in')
            else:
                self.mphys_add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem(self.name))
            self.connect('x_aero0','x_aero')

        self.mphys_add_pre_coupling_subsystem('aero', aero_builder, self.name)
        self.mphys_add_subsystem('coupling',aero_builder.get_coupling_group_subsystem(self.name))
        self.mphys_add_post_coupling_subsystem('aero', aero_builder, self.name)
