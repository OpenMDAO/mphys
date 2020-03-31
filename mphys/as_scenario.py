import openmdao.api as om
from omfsi.geo_disp import Geo_Disp

class AS_Scenario(om.Group):

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

        # get the elements from each builder
        aero = self.aero_builder.get_element()
        struct = self.struct_builder.get_element()
        disp_xfer, load_xfer = self.xfer_builder.get_element()

        # add the subgroups
        self.add_subsystem('disp_xfer', disp_xfer)
        self.add_subsystem('aero', aero)
        self.add_subsystem('load_xfer', load_xfer)
        self.add_subsystem('struct', struct)

        # set solvers
        self.nonlinear_solver=om.NonlinearBlockGS(maxiter=100)
        self.nonlinear_solver.options['iprint']=2
        self.linear_solver = om.LinearBlockGS(maxiter=100)

    def configure(self):

        # do the connections, this can be also done in setup
        self.connect('disp_xfer.u_a', 'aero.u_a')
        self.connect('aero.f_a', 'load_xfer.f_a')
        self.connect('load_xfer.f_s', 'struct.f_s')
        self.connect('struct.u_s', ['disp_xfer.u_s', 'load_xfer.u_s'])
