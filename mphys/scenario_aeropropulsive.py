import openmdao.api as om

from .coupling_group import CouplingGroup
from .scenario import Scenario


class ScenarioAeropropulsive(Scenario):
    def initialize(self):
        """
        A class to perform an aeropropulsive case.
        The Scenario will add the aerodynamic and propulsion builders' precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        super().initialize()

        self.options.declare("aero_builder", recordable=False, desc="The MPhys builder for the aerodynamic solver")
        self.options.declare("prop_builder", recordable=False, desc="The MPhys builder for the propulsion model")
        self.options.declare(
            "bc_coupling_builder",
            recordable=False,
            desc="The Mphys builder for the boundary condition coupling group",
            default=None,
        )
        self.options.declare(
            "in_MultipointParallel",
            default=False,
            desc="Set to `True` if adding this scenario inside a MultipointParallel Group.",
        )
        self.options.declare(
            "geometry_builder", default=None, recordable=False, desc="The optional Mphys builder for the geometry"
        )

    def _mphys_scenario_setup(self):
        aero_builder = self.options["aero_builder"]
        prop_builder = self.options["prop_builder"]
        bc_coupling_builder = self.options["bc_coupling_builder"]
        geometry_builder = self.options["geometry_builder"]

        if self.options["in_MultipointParallel"]:
            self._mphys_initialize_builders(aero_builder, prop_builder, geometry_builder)
            self._mphys_add_mesh_and_geometry_subsystems(aero_builder, geometry_builder)

        self._mphys_add_pre_coupling_subsystem_from_builder("aero", aero_builder, self.name)
        self._mphys_add_pre_coupling_subsystem_from_builder("prop", prop_builder, self.name)

        coupling_group = CouplingAeropropulsive(
            aero_builder=aero_builder, prop_builder=prop_builder, scenario_name=self.name
        )
        self.mphys_add_subsystem("coupling", coupling_group)

        self._mphys_add_post_coupling_subsystem_from_builder("prop", prop_builder, self.name)
        self._mphys_add_post_coupling_subsystem_from_builder("aero", aero_builder, self.name)

        if bc_coupling_builder is not None:
            self._mphys_add_post_coupling_subsystem_from_builder("bc_coupling", bc_coupling_builder, self.name)

    def _mphys_initialize_builders(self, aero_builder, prop_builder, geometry_builder):
        aero_builder.initialize(self.comm)
        prop_builder.initialize(self.comm)
        if geometry_builder is not None:
            geometry_builder.initialize(self.comm)

    def _mphys_add_mesh_and_geometry_subsystems(self, aero_builder, geometry_builder):

        if geometry_builder is None:
            self.mphys_add_subsystem("aero_mesh", aero_builder.get_mesh_coordinate_subsystem(self.name))
        else:
            self.add_subsystem("aero_mesh", aero_builder.get_mesh_coordinate_subsystem(self.name))
            self.mphys_add_subsystem("geometry", geometry_builder.get_mesh_coordinate_subsystem(self.name))
            self.connect("aero_mesh.x_aero0", "geometry.x_aero_in")

    def mphys_make_aeroprop_conn(self, aero2prop_conn, prop2aero_conn):
        # TODO automate this with mphys_result or mphys_coupling tags

        # make the connections
        for k, v in aero2prop_conn.items():
            self.connect("coupling.aero.%s" % k, "coupling.prop.%s" % v)
        for k, v in prop2aero_conn.items():
            self.connect("coupling.prop.%s" % k, "coupling.aero.%s" % v)


class CouplingAeropropulsive(CouplingGroup):
    """
    The standard aeropropulsive coupling problem.
    """

    def initialize(self):
        self.options.declare("aero_builder", recordable=False)
        self.options.declare("prop_builder", recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        aero_builder = self.options["aero_builder"]
        prop_builder = self.options["prop_builder"]
        scenario_name = self.options["scenario_name"]

        aero = aero_builder.get_coupling_group_subsystem(scenario_name)
        prop = prop_builder.get_coupling_group_subsystem(scenario_name)

        self.mphys_add_subsystem("aero", aero)
        self.mphys_add_subsystem("prop", prop)

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2, atol=1e-8, rtol=1e-8)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, atol=1e-8, rtol=1e-8)
