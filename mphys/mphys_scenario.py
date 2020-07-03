import openmdao.api as om
from mphys.mphys_solver_group import MPHYS_SolverGroup

class MPHYS_Scenario(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('builders', allow_none=False)
        self.options.declare('aero_discipline', allow_none=False)
        self.options.declare('struct_discipline', allow_none=False)
        self.options.declare('as_coupling', allow_none=False)

    def setup(self):

        # set flags
        self.aero_discipline = self.options['aero_discipline']
        self.struct_discipline = self.options['struct_discipline']
        self.as_coupling = self.options['as_coupling']

        # set the builders
        self.aero_builder = self.options['builders']['aero']
        self.struct_builder = self.options['builders']['struct']
        self.xfer_builder = self.options['builders']['xfer']

        # add the solver group itself and pass in the builders
        # this group will converge the nonlinear analysis
        self.add_subsystem(
            'solver_group',
            MPHYS_SolverGroup(
                builders=self.options['builders'],
                aero_discipline = self.aero_discipline,
                struct_discipline = self.struct_discipline,
                as_coupling = self.as_coupling
            )
        )

        # check if builders provide a scenario-level element.
        # e.g. a functionals component that is run once after
        # the nonlinear solver is converged.
        # we only check for disciplines, and we assume transfer
        # components do not have scenario level elements.
        if hasattr(self.aero_builder, 'get_scenario_element'):
            aero_scenario_element = self.aero_builder.get_scenario_element()
            self.add_subsystem('aero_funcs', aero_scenario_element)

            # if we have a scenario level element, we also need to
            # figure out what needs to be connected from the solver
            # level to the scenario level.
            scenario_conn = self.aero_builder.get_scenario_connections()
            # we can make these connections here
            for k, v in scenario_conn.items():
                self.connect('solver_group.aero.%s'%k, 'aero_funcs.%s'%v)

        # do the same for struct
        if hasattr(self.struct_builder, 'get_scenario_element'):
            struct_scenario_element = self.struct_builder.get_scenario_element()
            self.add_subsystem('struct_funcs', struct_scenario_element)

            scenario_conn = self.struct_builder.get_scenario_connections()
            for k, v in scenario_conn.items():
                self.connect('solver_group.struct.%s'%k, 'struct_funcs.%s'%v)

    def configure(self):
        pass