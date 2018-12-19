from openmdao.api import Group

class AeroSolverGroup(Group):
    """
    Aerodynamic solver
    """
    def initialize(self):
        self.options.declare('deformer',default=None,desc='CFD volume mesh deformer')
        self.options.declare('solver',default=None,desc='CFD flow solver')
        self.options.declare('force',default=None,desc='CFD surface force integration')

    def setup(self):
        self.add_subsystem('deformer',self.options['deformer'],promotes_inputs=['x_a0','x_a'],promotes_outputs=['x_g'])
        self.add_subsystem('solver',self.options['solver'],promotes_inputs=['dv_aero','x_g'],promotes_outputs=['q'])
        self.add_subsystem('force',self.options['force'],promotes_inputs=['x_g','q'],promotes_outputs=['f_a'])
