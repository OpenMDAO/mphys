import openmdao.api as om

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

        # get the elements from each builder
        if self.aero_discipline:
            aero = self.aero_builder.get_element(as_coupling=self.as_coupling)
        if self.struct_discipline:
            struct = self.struct_builder.get_element(as_coupling=self.as_coupling)
        if self.as_coupling:
            disp_xfer, load_xfer = self.xfer_builder.get_element()

        # add the subgroups
        if self.as_coupling:
            self.add_subsystem('disp_xfer', disp_xfer)
        if self.aero_discipline:
            self.add_subsystem('aero', aero)
        if self.as_coupling:
            self.add_subsystem('load_xfer', load_xfer)
        if self.struct_discipline:
            self.add_subsystem('struct', struct)

        # set solvers
        if self.as_coupling:
            self.nonlinear_solver=om.NonlinearBlockGS(maxiter=100)
            self.linear_solver = om.LinearBlockGS(maxiter=100)
            self.nonlinear_solver.options['iprint']=2

    def configure(self):

        # do the connections, this can be also done in setup
        if self.as_coupling:
            self.connect('disp_xfer.u_a', 'aero.u_a')
            self.connect('aero.f_a', 'load_xfer.f_a')
            self.connect('load_xfer.f_s', 'struct.f_s')
            self.connect('struct.u_s', ['disp_xfer.u_s', 'load_xfer.u_s'])
