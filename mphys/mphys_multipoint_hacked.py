from collections import OrderedDict
import openmdao.api as om
from mphys.mphys_scenario import MPHYS_Scenario
from mphys.mphys_error import MPHYS_Error

class MPHYS_Multipoint(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_builder', default=None)
        self.options.declare('struct_builder', default=None)
        self.options.declare('xfer_builder', default=None)

        # ordered dict to save all the scenarios user adds
        self.scenarios = OrderedDict()

    def setup(self):

        # set the builders
        self.aero_builder = self.options['aero_builder']
        self.struct_builder = self.options['struct_builder']
        self.xfer_builder = self.options['xfer_builder']

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

        # check if we have aero and structure
        if self.aero_discipline and self.struct_discipline:

            # aerostructural coupling is active
            self.as_coupling = True

            # we need a xfer builder for aero and structures
            if self.xfer_builder is None:
                raise MPHYS_Error('MPHYS_Multipoint group requires a transfer builder to couple aerodynamic and structural analyses.')

            # now initialize the xfer object
            self.xfer_builder.init_xfer_object(self.comm)
        else:
            # we dont have aerostructural coupling
            self.as_coupling = False

        # get the mesh elements from disciplines
        if self.aero_discipline:
            aero_mesh = self.aero_builder.get_mesh_element()
            #####self.add_subsystem('aero_mesh', aero_mesh)

        if self.struct_discipline:
            struct_mesh = self.struct_builder.get_mesh_element()
            #####self.add_subsystem('struct_mesh', struct_mesh)

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
                try:
                    for var in self.struct_builder.mesh_connections:
                        self.connect('struct_mesh.%s'% var,'%s.struct.%s' %(name,var))
                except:
                    #####self.connect('struct_mesh.x_s0', '%s.struct.x_s0'%name)
                    pass
                if self.as_coupling:
                    target_x_s0 =     ['%s.disp_xfer.x_s0'%name]
                    target_x_s0.append('%s.load_xfer.x_s0'%name)
                    #####self.connect('struct_mesh.x_s0', target_x_s0)

            if self.aero_discipline:
                try:
                    for var in self.aero_builder.mesh_connections:
                        self.connect('aero_mesh.%s'% var,'%s.aero.%s' %(name,var))
                except:
                    #####self.connect('aero_mesh.x_a0', '%s.aero.x_a0'%name)
                    pass
                if self.as_coupling:
                    target_x_a0 =     ['%s.load_xfer.x_a0'%name]
                    target_x_a0.append('%s.disp_xfer.x_a0'%name)
                    #####self.connect('aero_mesh.x_a0', target_x_a0)

    def mphys_add_scenario(self, name, **kwargs):
        # save all the data until we are ready to initialize the objects themselves
        self.scenarios[name] = kwargs

    def _mphys_add_scenario(self, name, min_procs=None, max_procs=None, aero_kwargs={}, struct_kwargs={}, xfer_kwargs={}):
        # this is the actual routine that does the addition of the OpenMDAO groups
        # this is called during the setup of this class
        self.add_subsystem(
            name,
            MPHYS_Scenario(
                builders = {
                    'aero': self.aero_builder,
                    'struct': self.struct_builder,
                    'xfer': self.xfer_builder,
                },
                aero_discipline = self.aero_discipline,
                struct_discipline = self.struct_discipline,
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
