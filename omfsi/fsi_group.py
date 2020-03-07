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
        self.nonlinear_solver.options['iprint']=2
        self.linear_solver = om.LinearBlockGS(maxiter=100)

    def configure(self):

        # add the i/o calls that have a size dependancy
        # this can also be done in the setup for this case since we have the solvers created already
        # but in general, this may not hold. On the other hand, during configure, every component
        # will be created, and we definitely can access this information.
        struct_ndof   = self.struct.get_ndof()
        struct_nnodes = self.struct.get_nnodes()
        aero_nnodes   = int(self.aero_solver.getSurfaceCoordinates().size /3)

        self.disp_xfer.add_io(struct_ndof, struct_nnodes, aero_nnodes)
        self.load_xfer.add_io(struct_ndof, struct_nnodes, aero_nnodes)
        self.geo_disp.add_io(aero_nnodes)

        # make connections
        self.connect('disp_xfer.u_a', 'geo_disp.u_a')
        self.connect('geo_disp.x_a', 'aero.deformer.x_a')
        self.connect('aero.force.f_a', 'load_xfer.f_a')
        self.connect('load_xfer.f_s', 'struct.f_s')
        self.connect('struct.u_s', ['disp_xfer.u_s', 'load_xfer.u_s'])

