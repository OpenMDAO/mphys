import openmdao.api as om

from mphys.core import Builder, CouplingGroup, MPhysVariables, Scenario
from mphys.scenarios.geo_disp import GeoDisp


class ScenarioAeroStructural(Scenario):
    def initialize(self):
        """
        A class to perform an aerodynamics + structures coupled case.
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
            aero_builder = self.options["aero_builder"]
            self.mphys_add_subsystem(
                "aero", aero_builder.get_coupling_group_subsystem(self.name)
            )

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
        aero_builder: Builder = self.options["aero_builder"]
        struct_builder: Builder = self.options["struct_builder"]
        geometry_builder: Builder = self.options["geometry_builder"]

        if geometry_builder is None:
            self.mphys_add_subsystem(
                "aero_mesh", aero_builder.get_mesh_coordinate_subsystem(self.name)
            )
            self.mphys_add_subsystem(
                "struct_mesh", struct_builder.get_mesh_coordinate_subsystem(self.name)
            )
            self.connect(
                MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES,
                MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
            )

            self.connect(
                MPhysVariables.Structures.Mesh.COORDINATES,
                MPhysVariables.Structures.COORDINATES,
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
            self.connect(
                f"aero_mesh.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}",
                MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT,
            )
            self.connect(
                MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT,
                MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL,
            )

            self.connect(
                f"struct_mesh.{MPhysVariables.Structures.Mesh.COORDINATES}",
                MPhysVariables.Structures.Geometry.COORDINATES_INPUT,
            )
            self.connect(
                MPhysVariables.Structures.Geometry.COORDINATES_OUTPUT,
                MPhysVariables.Structures.COORDINATES,
            )


class CouplingAeroStructural(CouplingGroup):
    """
    The standard aerostructural coupling problem.
    """

    def initialize(self):
        self.options.declare("aero_builder", recordable=False)
        self.options.declare("struct_builder", recordable=False)
        self.options.declare("ldxfer_builder", recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        aero_builder: Builder = self.options["aero_builder"]
        struct_builder: Builder = self.options["struct_builder"]
        ldxfer_builder: Builder = self.options["ldxfer_builder"]
        scenario_name = self.options["scenario_name"]

        disp_xfer, load_xfer = ldxfer_builder.get_coupling_group_subsystem(
            scenario_name
        )
        aero = aero_builder.get_coupling_group_subsystem(scenario_name)
        struct = struct_builder.get_coupling_group_subsystem(scenario_name)

        geo_disp = GeoDisp(number_of_nodes=aero_builder.get_number_of_nodes())

        self.mphys_add_subsystem("disp_xfer", disp_xfer)
        self.mphys_add_subsystem("geo_disp", geo_disp)
        self.mphys_add_subsystem("aero", aero)
        self.mphys_add_subsystem("load_xfer", load_xfer)
        self.mphys_add_subsystem("struct", struct)

        self.nonlinear_solver = om.NonlinearBlockGS(
            maxiter=25, iprint=2, atol=1e-8, rtol=1e-8, use_aitken=True
        )
        self.linear_solver = om.LinearBlockGS(
            maxiter=25, iprint=2, atol=1e-8, rtol=1e-8, use_aitken=True
        )
