from collections import OrderedDict
import openmdao.api as om
from mphys.scenario import Scenario
from mphys.error import MPHYS_Error

class Multipoint(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_builder', default=None)
        self.options.declare('struct_builder', default=None)
        self.options.declare('xfer_builder', default=None)
        self.options.declare('prop_builder', default=None)

        # ordered dict to save all the scenarios user adds
        self.scenarios = OrderedDict()

    def setup(self):

        # set the builders
        self.aero_builder = self.options['aero_builder']
        self.struct_builder = self.options['struct_builder']
        self.xfer_builder = self.options['xfer_builder']
        self.prop_builder = self.options['prop_builder']

        # we need to initialize the aero and struct objects before the xfer one
        # potentially need to do fancy stuff with the comms here if user wants to run in parallel
        # in that case, the scenarios themselves would likely need to initialize solvers themselves

        # only initialize solvers if we have the builders
        if self.aero_builder is not None:
            self.aero_builder.init_solver(self.comm)
            self.aero_discipline = True
        else:
            # no aero builder, so we won't have this discipline
            self.aero_discipline = False

        if self.struct_builder is not None:
            self.struct_builder.init_solver(self.comm)
            self.struct_discipline = True
        else:
            # no struct builder, so we won't have this discipline
            self.struct_discipline = False

        if self.prop_builder is not None:
            self.prop_builder.init_solver(self.comm)
            self.prop_discipline = True
        else:
            self.prop_discipline = False

        # check if we have aero and structure
        if self.aero_discipline and self.struct_discipline:

            # aerostructural coupling is active
            self.as_coupling = True

            # we need a xfer builder for aero and structures
            if self.xfer_builder is None:
                raise MPHYS_Error('Multipoint group requires a transfer builder to couple aerodynamic and structural analyses.')

            # now initialize the xfer object
            self.xfer_builder.init_xfer_object(self.comm)
        else:
            # we dont have aerostructural coupling
            self.as_coupling = False

        # get the mesh elements from disciplines
        if self.aero_discipline:
            aero_mesh = self.aero_builder.get_mesh_element()
            self.add_subsystem('aero_mesh', aero_mesh)

        if self.struct_discipline:
            struct_mesh = self.struct_builder.get_mesh_element()
            self.add_subsystem('struct_mesh', struct_mesh)

        # add openmdao groups for each scenario
        for name, kwargs in self.scenarios.items():
            self._mphys_add_scenario(name, **kwargs)

        # set solvers
        self.nonlinear_solver = om.NonlinearRunOnce()
        self.linear_solver    = om.LinearRunOnce()

    def configure(self):
        # connect the initial mesh coordinates.
        # at this stage, everything is allocated and every group/component
        # below this level is set up.

        # make the default and custom connections for the struct discipline
        if self.struct_discipline:

            # get if we have any connections to the solver level groups
            solver_connections = self.struct_mesh.get_io_metadata(iotypes='output', tags='solver')

            # similarly for functionals
            funcs_connections = self.struct_mesh.get_io_metadata(iotypes='output', tags='funcs')

            # now loop over scenarios and connect
            for scenario in self.scenarios:
                # first, make the default mesh connections for as_coupling
                if self.as_coupling:
                    target_x_s0 =     ['%s.solver_group.disp_xfer.x_s0'%scenario]
                    target_x_s0.append('%s.solver_group.load_xfer.x_s0'%scenario)
                    self.connect('struct_mesh.x_s0', target_x_s0)

                # then connect the custom stuff to solver and functionals
                for k, v in solver_connections.items():
                    # get the variable name
                    var_name = k.split('.')[-1]
                    # we assume this variable exists for connection in the solver group
                    # and is promoted to the group level.
                    self.connect('struct_mesh.%s'%v['prom_name'], '%s.solver_group.struct.%s'%(scenario, var_name))

                # same deal for functionals
                for k, v in funcs_connections.items():
                    var_name = k.split('.')[-1]
                    self.connect('struct_mesh.%s'%v['prom_name'], '%s.struct_funcs.%s'%(scenario, var_name))

        # do the same for the aero discipline
        if self.aero_discipline:

            # get if we have any connections to the solver level groups
            solver_connections = self.aero_mesh.get_io_metadata(iotypes='output', tags='solver')

            # similarly for functionals
            funcs_connections = self.aero_mesh.get_io_metadata(iotypes='output', tags='funcs')

            # now loop over scenarios and connect
            for scenario in self.scenarios:
                # first, make the default mesh connections for as_coupling
                if self.as_coupling:
                    target_x_a0 =     ['%s.solver_group.load_xfer.x_a0'%scenario]
                    target_x_a0.append('%s.solver_group.disp_xfer.x_a0'%scenario)
                    self.connect('aero_mesh.x_a0', target_x_a0)

                # then connect the custom stuff to solver and functionals
                for k, v in solver_connections.items():
                    # get the variable name
                    var_name = k.split('.')[-1]
                    # we assume this variable exists for connection in the solver group
                    # and is promoted to the group level.
                    self.connect('aero_mesh.%s'%v['prom_name'], '%s.solver_group.aero.%s'%(scenario, var_name))

                # same deal for functionals
                for k, v in funcs_connections.items():
                    var_name = k.split('.')[-1]
                    self.connect('aero_mesh.%s'%v['prom_name'], '%s.aero_funcs.%s'%(scenario, var_name))

    def mphys_add_scenario(self, name, **kwargs):
        # save all the data until we are ready to initialize the objects themselves
        self.scenarios[name] = kwargs

    def _mphys_add_scenario(self, name, min_procs=None, max_procs=None, aero_kwargs={}, struct_kwargs={}, xfer_kwargs={}):
        # this is the actual routine that does the addition of the OpenMDAO groups
        # this is called during the setup of this class
        self.add_subsystem(
            name,
            Scenario(
                builders = {
                    'aero': self.aero_builder,
                    'struct': self.struct_builder,
                    'prop': self.prop_builder,
                    'xfer': self.xfer_builder,
                },
                aero_discipline = self.aero_discipline,
                struct_discipline = self.struct_discipline,
                prop_discipline = self.prop_discipline,
                as_coupling = self.as_coupling
            )
        )

    def mphys_add_coordinate_input(self):
        # add the coordinates as inputs for every discipline we have
        points = {}

        if self.aero_discipline:
            name, x_a0 = self.aero_mesh.mphys_add_coordinate_input()
            points['aero_points'] = x_a0
            self.promotes('aero_mesh', inputs=[(name, 'aero_points')])

        if self.struct_discipline:
            name, x_s0 = self.struct_mesh.mphys_add_coordinate_input()
            points['struct_points'] = x_s0
            self.promotes('struct_mesh', inputs=[(name, 'struct_points')])

        return points


    def mphys_get_triangulated_surface(self):
        # get triangulated surface for computing constraints
        if self.aero_discipline:
            x_a0_tri = self.aero_mesh.mphys_get_triangulated_surface()
            return x_a0_tri
        else:
            raise NotImplementedError('Only ADFlow format supported so far')
