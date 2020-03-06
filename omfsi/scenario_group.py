import openmdao.api as om
from omfsi.fsi_group import fsi_group
from omfsi.tacs_component_configure import TacsFunctions, TacsMass
from omfsi.adflow_component_configure import AdflowFunctions

class omfsi_scenario(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_solver', allow_none=False)
        self.options.declare('struct_solver', allow_none=False)
        self.options.declare('struct_objects', allow_none=False)
        self.options.declare('xfer_object', allow_none=False)

    def setup(self):

        # get all the initialized objects for computations
        self.aero_solver = self.options['aero_solver']
        self.struct_solver = self.options['struct_solver']
        self.struct_objects = self.options['struct_objects']
        self.xfer_object = self.options['xfer_object']

        # create the components and groups
        self.add_subsystem('fsi_group', fsi_group(
            aero_solver=self.aero_solver,
            struct_solver=self.struct_solver,
            struct_objects=self.struct_objects,
            xfer_object=self.xfer_object
        ))
        self.add_subsystem('struct_funcs', TacsFunctions(
            struct_solver=self.struct_solver
        ))
        self.add_subsystem('struct_mass', TacsMass(
            struct_solver=self.struct_solver
        ))
        self.add_subsystem('aero_funcs', AdflowFunctions(
            aero_solver=self.aero_solver
        ))

        # set solvers
        self.nonlinear_solver=om.NonlinearRunOnce()
        self.linear_solver = om.LinearRunOnce()

    # def configure(self):

        # now we need to connect all these components...

