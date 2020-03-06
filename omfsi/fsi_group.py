import openmdao.api as om

from omfsi.tacs_component_configure import TacsSolver
from omfsi.adflow_component_configure import AdflowGroup, GeoDisp
from omfsi.meld_xfer_component_configure import MeldDisplacementTransfer, MeldLoadTransfer

class fsi_group(om.Group):

    def initialize(self):

        self.options.declare('aero_solver', allow_none=False)
        self.options.declare('struct_solver', allow_none=False)
        self.options.declare('struct_objects', allow_none=False)
        self.options.declare('xfer_object', allow_none=False)

    def setup(self):

        self.aero_solver = self.options['aero_solver']
        self.struct_solver = self.options['struct_solver']
        self.struct_objects = self.options['struct_objects']
        self.xfer_object = self.options['xfer_object']

        self.add_subsystem('disp_xfer', MeldDisplacementTransfer(
            xfer_object=self.xfer_object
        ))
        self.add_subsystem('geo_disp', GeoDisp())
        self.add_subsystem('aero', AdflowGroup(
            aero_solver=self.aero_solver
        ))
        self.add_subsystem('load_xfer', MeldLoadTransfer(
            xfer_object=self.xfer_object
        ))
        self.add_subsystem('struct', TacsSolver(
            struct_solver=self.struct_solver,
            struct_objects=self.struct_objects
        ))

        # set solvers
        self.nonlinear_solver=om.NonlinearBlockGS(maxiter=100)
        self.linear_solver = om.LinearBlockGS(maxiter=100)

    # def configure(self):

        # now we need to connect all these components...
