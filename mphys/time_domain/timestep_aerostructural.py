from mphys.coupling_aerostructural import CouplingAeroStructural
from .timestep import TimeStep


class TimeStepAeroStructural(TimeStep):
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

    def _mphys_timestep_setup(self):
        aero_builder = self.options["aero_builder"]
        struct_builder = self.options["struct_builder"]
        ldxfer_builder = self.options["ldxfer_builder"]
        builders = [aero_builder, struct_builder, ldxfer_builder]

        self._add_ivc_with_mphys_inputs(builders, self.options["user_input_variables"])
        self._add_ivc_with_state_backplanes(builders)
        self._add_ivc_with_time_information()

        self._mphys_add_pre_coupling_subsystem_from_builder("aero", aero_builder)
        self._mphys_add_pre_coupling_subsystem_from_builder("struct", struct_builder)
        self._mphys_add_pre_coupling_subsystem_from_builder("ldxfer", ldxfer_builder)

        coupling_group = CouplingAeroStructural(
            aero_builder=aero_builder,
            struct_builder=struct_builder,
            ldxfer_builder=ldxfer_builder,
            scenario_name=self.name,
        )
        self.mphys_add_subsystem("coupling", coupling_group)

        self._mphys_add_post_coupling_subsystem_from_builder("ldxfer", ldxfer_builder)
        self._mphys_add_post_coupling_subsystem_from_builder("aero", aero_builder)
        self._mphys_add_post_coupling_subsystem_from_builder("struct", struct_builder)
