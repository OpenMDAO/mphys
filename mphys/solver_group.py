import openmdao.api as om

class SolverGroup(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('builders', allow_none=False, recordable=False)
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

        # get the elements from each builder
        if self.aero_discipline:
            aero = self.aero_builder.get_element(as_coupling=self.as_coupling)
        if self.struct_discipline:
            struct = self.struct_builder.get_element(as_coupling=self.as_coupling)
        if self.as_coupling:
            disp_xfer, load_xfer = self.xfer_builder.get_element()

        if self.prop_discipline:
            prop = self.prop_builder.get_element()

        # add the subgroups
        if self.as_coupling:
            self.add_subsystem('disp_xfer', disp_xfer)
        if self.aero_discipline:
            self.add_subsystem('aero', aero)
        if self.as_coupling:
            self.add_subsystem('load_xfer', load_xfer)
        if self.struct_discipline:
            self.add_subsystem('struct', struct)
        if self.prop_discipline:
            self.add_subsystem('prop', prop)

        # set solvers
        # TODO add a nonlinear solver when we have feedback coupling to prop
        if self.as_coupling:
            self.nonlinear_solver=om.NonlinearBlockGS(maxiter=50, iprint=2, atol=1e-8, rtol=1e-8, use_aitken=True)
            self.linear_solver = om.LinearBlockGS(maxiter=50, iprint =2, atol=1e-8,rtol=1e-8)

    def configure(self):

        # do the connections, this can be also done in setup
        if self.as_coupling:
            self.connect('disp_xfer.u_aero', 'aero.u_aero')
            self.connect('aero.f_aero', 'load_xfer.f_aero')
            self.connect('load_xfer.f_struct', 'struct.f_struct')
            self.connect('struct.u_struct', ['disp_xfer.u_struct', 'load_xfer.u_struct'])
