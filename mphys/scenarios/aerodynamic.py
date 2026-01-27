from mphys.core import Builder, MPhysVariables, Scenario


class ScenarioAerodynamic(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        super().initialize()

        self.options.declare(
            "aero_builder",
            recordable=False,
            desc="The MPhys builder for the aerodynamic solver",
        )
        self.options.declare(
            "in_MultipointParallel",
            default=False,
            types=bool,
            desc="Set to `True` if adding this scenario inside a MultipointParallel Group.",
        )
        self.options.declare(
            "geometry_builder",
            default=None,
            recordable=False,
            desc="The optional MPhys builder for the geometry",
        )

    def _mphys_scenario_setup(self):
        aero_builder: Builder = self.options["aero_builder"]
        geometry_builder: Builder = self.options["geometry_builder"]

        if self.options["in_MultipointParallel"]:
            aero_builder.initialize(self.comm)

            if geometry_builder is not None:
                geometry_builder.initialize(self.comm)
                self.mphys_add_subsystem(
                    "mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
                )
                self.mphys_add_subsystem(
                    "geometry",
                    geometry_builder.get_mesh_coordinate_subsystem(self.name),
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT,
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES,
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
                )
            else:
                self.mphys_add_subsystem(
                    "mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES,
                )
                self.connect(
                    MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                    MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
                )

        self._mphys_add_pre_coupling_subsystem_from_builder(
            "aero", aero_builder, self.name
        )
        self.mphys_add_subsystem(
            "coupling", aero_builder.get_coupling_group_subsystem(self.name)
        )
        self._mphys_add_post_coupling_subsystem_from_builder(
            "aero", aero_builder, self.name
        )
