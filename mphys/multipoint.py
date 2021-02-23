from collections import OrderedDict
import openmdao.api as om
from mphys.scenario import Scenario
from mphys.error import MPHYS_Error

class Multipoint(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_builder', default=None, recordable=False)
        self.options.declare('struct_builder', default=None, recordable=False)
        self.options.declare('xfer_builder', default=None, recordable=False)
        self.options.declare('prop_builder', default=None, recordable=False)

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

        # loop over scenarios and connect them all
        for name in self.scenarios:
            if self.struct_discipline:
                # make the default mesh connections for as_coupling
                if self.as_coupling:
                    #TODO: remove this once all elements can use src_indices at this level or use shape_by_connection
                    #target_x_s0 =     ['%s.solver_group.disp_xfer.x_struct0'%name]
                    #target_x_s0.append('%s.solver_group.load_xfer.x_struct0'%name)
                    #self.connect('struct_mesh.x_struct0', target_x_s0)
                    target_x_s0 =     ['%s.solver_group.disp_xfer.x_struct0'%name]
                    scenario = getattr(self, name)
                    self.connect('struct_mesh.x_struct0', target_x_s0, src_indices=scenario.solver_group.disp_xfer.src_indices['x_struct0'])

                    target_x_s0 = ['%s.solver_group.load_xfer.x_struct0'%name]
                    self.connect('struct_mesh.x_struct0', target_x_s0)

                # check if we have custom mesh connections
                if hasattr(self.struct_builder, 'get_mesh_connections'):
                    mesh_conn = self.struct_builder.get_mesh_connections()

                    # if mesh_conn has entries called 'solver' or 'funcs',
                    # then we know that these are dictionaries of connections
                    # to be made to solver or funcs. If there are no solver
                    # or funcs entries in here, we just assume every key
                    # will be connected to the solver.
                    if ('solver' in mesh_conn) or ('funcs' not in mesh_conn):
                        # if solver is in the dict, we connect the keys of that dict
                        if 'solver' in mesh_conn:
                            mesh_to_solver = mesh_conn['solver']
                        # if solver is not in this, it means that funcs is not in
                        # mesh_conn, which then means that mesh_conn only has
                        # connections that go to the solver by default
                        else:
                            mesh_to_solver = mesh_conn

                        for k,v in mesh_to_solver.items():
                            self.connect('struct_mesh.%s'%k, '%s.solver_group.struct.%s'%(name, v))

                    # if funcs is in the dict, we just connect the entries from this to the funcs
                    if 'funcs' in mesh_conn:
                        mesh_to_funcs = mesh_conn['funcs']
                        for k,v in mesh_to_funcs.items():
                            self.connect('struct_mesh.%s'%k, '%s.struct_funcs.%s'%(name, v))

                # if the solver did not define any custom mesh connections,
                # we will just connect the nodes from the mesh to solver
                else:
                    self.connect('struct_mesh.x_struct0', '%s.solver_group.struct.x_struct0'%name)

            if self.aero_discipline:
                # make the default mesh connections for as_coupling
                if self.as_coupling:
                    #TODO: remove this once all elements can use src_indices at this level or use shape_by_connection
                    #target_x_a0 =     ['%s.solver_group.load_xfer.x_aero0'%name]
                    #target_x_a0.append('%s.solver_group.disp_xfer.x_aero0'%name)
                    #self.connect('aero_mesh.x_aero0', target_x_a0)

                    target_x_a0 =     ['%s.solver_group.disp_xfer.x_aero0'%name]
                    scenario = getattr(self, name)
                    self.connect('aero_mesh.x_aero0', target_x_a0, src_indices=scenario.solver_group.disp_xfer.src_indices['x_aero0'])

                    target_x_a0 =     ['%s.solver_group.load_xfer.x_aero0'%name]
                    self.connect('aero_mesh.x_aero0', target_x_a0)


                # check if we have custom mesh connections
                if hasattr(self.aero_builder, 'get_mesh_connections'):
                    mesh_conn = self.aero_builder.get_mesh_connections()

                    # if mesh_conn has entries called 'solver' or 'funcs',
                    # then we know that these are dictionaries of connections
                    # to be made to solver or funcs. If there are no solver
                    # or funcs entries in here, we just assume every key
                    # will be connected to the solver.
                    if ('solver' in mesh_conn) or ('funcs' not in mesh_conn):
                        # if solver is in the dict, we connect the keys of that dict
                        if 'solver' in mesh_conn:
                            mesh_to_solver = mesh_conn['solver']
                        # if solver is not in this, it means that funcs is not in
                        # mesh_conn, which then means that mesh_conn only has
                        # connections that go to the solver by default
                        else:
                            mesh_to_solver = mesh_conn

                        for k,v in mesh_to_solver.items():
                            self.connect('aero_mesh.%s'%k, '%s.solver_group.aero.%s'%(name, v))

                    # if funcs is in the dict, we just connect the entries from this to the funcs
                    if 'funcs' in mesh_conn:
                        mesh_to_funcs = mesh_conn['funcs']
                        for k,v in mesh_to_funcs.items():
                            self.connect('aero_mesh.%s'%k, '%s.aero_funcs.%s'%(name, v))

                # if the solver did not define any custom mesh connections,
                # we will just connect the nodes from the mesh to solver
                else:
                    self.connect('aero_mesh.x_aero0', '%s.solver_group.aero.x_aero0'%name)


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
