import openmdao.api as om
from mphys.solver_group import SolverGroup

class Scenario(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('builders', allow_none=False)
        self.options.declare('aero_discipline', allow_none=False)
        self.options.declare('struct_discipline', allow_none=False)
        self.options.declare('prop_discipline', allow_none=False)
        self.options.declare('as_coupling', allow_none=False)

    def setup(self):

        # set flags
        self.aero_discipline = self.options['aero_discipline']
        self.struct_discipline = self.options['struct_discipline']
        self.prop_discipline = self.options['prop_discipline']
        self.as_coupling = self.options['as_coupling']

        # set the builders
        self.aero_builder = self.options['builders']['aero']
        self.struct_builder = self.options['builders']['struct']
        self.prop_builder = self.options['builders']['prop']
        self.xfer_builder = self.options['builders']['xfer']

        # add the solver group itself and pass in the builders
        # this group will converge the nonlinear analysis
        self.add_subsystem(
            'solver_group',
            SolverGroup(
                builders=self.options['builders'],
                aero_discipline = self.aero_discipline,
                struct_discipline = self.struct_discipline,
                prop_discipline = self.prop_discipline,
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

        # do the same for struct
        if hasattr(self.struct_builder, 'get_scenario_element'):
            struct_scenario_element = self.struct_builder.get_scenario_element()
            self.add_subsystem('struct_funcs', struct_scenario_element)

            scenario_conn = self.struct_builder.get_scenario_connections()
            for k, v in scenario_conn.items():
                self.connect('solver_group.struct.%s'%k, 'struct_funcs.%s'%v)

    def configure(self):

        # if we have a scenario level element for a disciplinary solver, we need to check if we have any
        # connections between the solver level element and scenario level element

        # aero
        if self.aero_discipline:
            # check if solver has any connections for funcs. this dictionary will be empty
            # if we don't have any connections to the functionals group. If the functionals group
            # do not exist, the dictionary should still be empty, so we don't check.
            funcs_connections = self.solver_group.aero.get_io_metadata(iotypes='output', tags='funcs')

            # connect these to the functionals group
            for k, v in funcs_connections.items():
                # get the variable name
                var_name = k.split('.')[-1]
                # we assume this variable exists for connection in the functional group
                # # and is promoted to the functional group level.
                self.connect('solver_group.aero.%s'%v['prom_name'], 'aero_funcs.%s'%var_name)

        if self.struct_discipline:
            # struct, see above for comments
            funcs_connections = self.solver_group.struct.get_io_metadata(iotypes='output', tags='funcs')
            for k, v in funcs_connections.items():
                var_name = k.split('.')[-1]
                self.connect('solver_group.struct.%s'%v['prom_name'], 'struct_funcs.%s'%var_name)

    def mphys_make_aeroprop_conn(self, aero2prop_conn, prop2aero_conn):
        # do the connections. we may want to do the solver level connections on the solver level but do them here for now
        for k,v in aero2prop_conn.items():
            self.connect('solver_group.aero.%s'%k, 'solver_group.prop.%s'%v)

        # also do the connections from prop to aero...
        for k,v in prop2aero_conn.items():
            self.connect('solver_group.prop.%s'%k, 'solver_group.aero.%s'%v)
