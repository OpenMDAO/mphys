from mphys.scenario import Scenario
from mphys.coupling_aerostructural import CouplingAeroStructural


class ScenarioAeroStructural(Scenario):
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
            "struct_builder",
            recordable=False,
            desc="The MPhys builder for the structural solver",
        )
        self.options.declare(
            "ldxfer_builder",
            recordable=False,
            desc="The MPhys builder for the load and displacement transfer",
        )
        self.options.declare(
            "in_MultipointParallel",
            default=False,
            desc="Set to `True` if adding this scenario inside a MultipointParallel Group.",
        )
        self.options.declare(
            "geometry_builder",
            default=None,
            recordable=False,
            desc="The optional MPhys builder for the geometry",
        )
        self.options.declare(
            "coupling_group_type",
            default="full_coupling",
            desc='Limited flexibility for coupling group type to accomodate flutter about jig shape or DLM where coupling group can be skipped: ["full_coupling", "aerodynamics_only", None]',
        )
        self.options.declare(
            "pre_coupling_order",
            default=["aero", "struct", "ldxfer"],
            recordable=False,
            desc="The order of the pre coupling subsystems",
        )
        self.options.declare(
            "post_coupling_order",
            default=["ldxfer", "aero", "struct"],
            recordable=False,
            desc="The order of the post coupling subsystems",
        )

    def _mphys_scenario_setup(self):
        if self.options["in_MultipointParallel"]:
            self._mphys_initialize_builders()
            self._mphys_add_mesh_and_geometry_subsystems()

        self._mphys_add_pre_coupling_subsystems()
        self._mphys_add_coupling_group()
        self._mphys_add_post_coupling_subsystems()

    def _mphys_check_coupling_order_inputs(self, given_options):
        valid_options = ["aero", "struct", "ldxfer"]

        length = len(given_options)
        if length > 3:
            raise ValueError(
                f"Specified too many items in the pre/post coupling order list, len={length}"
            )

        for option in given_options:
            if option not in valid_options:
                raise ValueError(
                    f"""Unknown pre/post order option: {option}. valid options are ["{'", "'.join(valid_options)}"]"""
                )

    def _mphys_add_pre_coupling_subsystems(self):
        self._mphys_check_coupling_order_inputs(self.options["pre_coupling_order"])
        for discipline in self.options["pre_coupling_order"]:
            self._mphys_add_pre_coupling_subsystem_from_builder(
                discipline, self.options[f"{discipline}_builder"], self.name
            )

    def _mphys_add_coupling_group(self):
        if self.options["coupling_group_type"] == "full_coupling":
            coupling_group = CouplingAeroStructural(
                aero_builder=self.options["aero_builder"],
                struct_builder=self.options["struct_builder"],
                ldxfer_builder=self.options["ldxfer_builder"],
                scenario_name=self.name,
            )
            self.mphys_add_subsystem("coupling", coupling_group)

        elif self.options["coupling_group_type"] == "aerodynamics_only":
            aero = self.options["aero_builder"].get_coupling_group_subsystem(self.name)
            self.mphys_add_subsystem("aero", aero)

    def _mphys_add_post_coupling_subsystems(self):
        self._mphys_check_coupling_order_inputs(self.options["post_coupling_order"])
        for discipline in self.options["post_coupling_order"]:
            self._mphys_add_post_coupling_subsystem_from_builder(
                discipline, self.options[f"{discipline}_builder"], self.name
            )

    def _mphys_initialize_builders(self):
        self.options["aero_builder"].initialize(self.comm)
        self.options["struct_builder"].initialize(self.comm)
        self.options["ldxfer_builder"].initialize(self.comm)

        geometry_builder = self.options["geometry_builder"]
        if geometry_builder is not None:
            geometry_builder.initialize(self.comm)

    def _mphys_add_mesh_and_geometry_subsystems(self):
        aero_builder = self.options["aero_builder"]
        struct_builder = self.options["struct_builder"]
        geometry_builder = self.options["geometry_builder"]

        if geometry_builder is None:
            self.mphys_add_subsystem(
                "aero_mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
            )
            self.mphys_add_subsystem(
                "struct_mesh", struct_builder.get_mesh_coordinate_subsystem(self.name)
            )
        else:
            self.add_subsystem(
                "aero_mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
            )
            self.add_subsystem(
                "struct_mesh", struct_builder.get_mesh_coordinate_subsystem(self.name)
            )
            self.mphys_add_subsystem(
                "geometry", geometry_builder.get_mesh_coordinate_subsystem(self.name)
            )
            self.connect("aero_mesh.x_aero0", "geometry.x_aero_in")
            self.connect("struct_mesh.x_struct0", "geometry.x_struct_in")
