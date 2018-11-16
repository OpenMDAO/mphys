from openmdao.api import ExplicitComponent, Group, NonlinearBlockGS, LinearBlockGS

class FsiSolver(Group):
    """
    FSI solver

    Variables:
        x_a0 = aerodynamic surface with geom changes
        u_a  = aerodynamic surface displacements
        x_a  = deformed aerodynamic surface
        f_a  = aerodynamic forces

        x_s0 = structural mesh with geom changes
        u_s  = structural mesh displacements
        f_s  = structural forces
    """
    def initialize(self):
        self.options.declare('aero'     ,default=None,desc='aerodynamic solver group')
        self.options.declare('struct'   ,default=None,desc='structural solver group')
        self.options.declare('disp_xfer',default=None,desc='displacement transfer')
        self.options.declare('load_xfer',default=None,desc='displacement transfer')

        self.options['distributed'] = True

    def setup(self):
        self.add_subsystem('aero',  self.options['aero'],  promotes_inputs=['x_a0','x_a','u_s'],promotes_outputs=['x_g','q','f_a'])
        self.add_subsystem('struct',self.options['struct'],promotes_inputs=['struct_dv','x_s0','f_s'],      promotes_outputs=['u_s'])


        self.add_subsystem('disp_xfer',self.options['disp_xfer'],promotes_inputs=['x_a0','x_s0','u_s'],promotes_outputs=['u_a'])
        self.add_subsystem('load_xfer',self.options['load_xfer'],promotes_inputs=['x_a0','x_s0','f_a'],promotes_outputs=['f_s'])

        # geo_disp does x_a0 + u_a
        self.add_subsystem('geo_disps',GeoDisp(),promotes_inputs=['x_a0','u_a'],promotes_outputs=['x_a'])

        self.nonlinear_solver = NonlinearBlockGS()
        self.linear_solver = LinearBlockGS()

class GeoDisp(ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        self.add_input('x_a0',size=self.size,desc='aerodynamic surface with geom change')
        self.add_input('u_a',size=self.size,desc='aerodynamic surface displacements')

        self.add_output('x_a',size=self.size,desc='deformed aerodynamic surface')
    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            d_outputs['x_a'] += d_inputs['x_a0'] + d_inputs['u_a']
        if mode == 'rev':
            d_inputs['x_a0'] += d_outputs['x_a']
            d_inputs['u_a']  += d_outputs['x_a']

