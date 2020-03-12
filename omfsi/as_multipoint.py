from collections import OrderedDict
import openmdao.api as om
from omfsi.as_scenario import AS_Scenario

class AS_Multipoint(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_builder', allow_none=False)
        self.options.declare('struct_builder', allow_none=False)
        self.options.declare('xfer_builder', allow_none=False)

    def setup(self):

        # set the builders
        self.aero_builder = self.options['aero_builder']
        self.struct_builder = self.options['struct_builder']
        self.xfer_builder = self.options['xfer_builder']

        # we need to initialize the aero and struct objects before the xfer one
        # potentially need to do fancy stuff with the comms here if user wants to run in parallel
        # in that case, the scenarios themselves would likely need to initialize solvers themselves
        self.aero_builder.init_solver(self.comm)
        self.struct_builder.init_solver(self.comm)
        self.xfer_builder.init_xfer_object(self.comm)

        # add openmdao groups for each scenario
        for name, kwargs in self.scenarios.items():
            self._mphy_add_scenario(name, **kwargs)

        # set solvers
        self.nonlinear_solver = om.NonlinearRunOnce()
        self.linear_solver    = om.LinearRunOnce()

    def configure(self):

        # connect the initial mesh coordinates.
        # with the configure-based approach, we do not need to have
        # separate components just to carry the initial mesh coordinates,
        # but we can directly pass them to all of the components here.
        # at this stage, everything is allocated and every group/component
        # below this level is set up.

        # loop over scenarios and connect them all
        n_scenario = self.options['n_scenario']
        for i in range(n_scenario):
            target_x_s0 = [
                'cruise%d.fsi_group.disp_xfer.x_s0'%i,
                'cruise%d.fsi_group.load_xfer.x_s0'%i,
                'cruise%d.fsi_group.struct.x_s0'%i,
                'cruise%d.struct_funcs.x_s0'%i,
                'cruise%d.struct_mass.x_s0'%i
            ]
            self.connect('struct_mesh.x_s0_mesh', target_x_s0)

            target_x_a0 = [
                'cruise%d.fsi_group.disp_xfer.x_a0'%i,
                'cruise%d.fsi_group.geo_disp.x_a0'%i,
                'cruise%d.fsi_group.load_xfer.x_a0'%i
            ]
            self.connect('aero_mesh.x_a0_mesh', target_x_a0)

    def mphy_add_scenario(self, name, **kwargs):
        # save all the inputs here until we are ready to do the initialization

        # create the dict if we haven't done already
        if self.scenarios is None:
            # we want to maintain the order of addition
            self.scenarios = OrderedDict()

        # save all the data until we are ready to initialize the objects themselves
        self.scenarios[name] = kwargs

    def _mphy_add_scenario(self, name, min_procs=None, max_procs=None, aero_kwargs={}, struct_kwargs={}, xfer_kwargs={}):
        # this is the actual routine that does the addition of the OpenMDAO groups
        # this is called during the setup of this class
        self.add_subsystem(
            s_name,
            AS_Scenario(
                aero_builder   = self.aero_builder,
                struct_builder = self.struct_builder,
                xfer_builder   = self.xfer_builder,
            )
            # also set the min/max procs here
        )
