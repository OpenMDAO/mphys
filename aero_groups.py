from openmdao.api import Group
from openmdao.api import NonlinearBlockGS

class AeroSolver(Group)
    """
    Aerodynamic solver
    """
    def initialize(self):
        self.options.declare('deformer',default=None,desc='CFD volume mesh deformer')
        self.options.declare('flow',default=None,desc='CFD flow solver')
        self.options.declare('force',default=None,desc='CFD surface force integration')

        self.options['distributed'] = True

    def setup(self):
        self.add_subsystem('deformer',self.options['deformer'],promotes_inputs=['x_a0','x_a'],promotes_outputs=['q'])
        self.add_subsystem('flow',self.options['flow'],promotes_inputs=['x_g'],promotes_outputs=['q'])
        self.add_subsystem('force',self.options['force'],promotes_inputs=['x_g','q'],promotes_outputs=['f_a'])
