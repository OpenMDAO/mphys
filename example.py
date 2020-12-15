import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        # set up model hierarchy
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.0)
        indeps.add_output('z', np.array([5.0, 2.0]))

        cycle = self.add_subsystem('cycle', om.Group())
        cycle.add_subsystem('d1', SellarDis1())
        cycle.add_subsystem('d2', SellarDis2())

        cycle.nonlinear_solver = om. NonlinearBlockGS()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0))

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

    def configure(self):
        # connect everything via promotes
        self.cycle.promotes('d1', inputs=['x', 'z', 'y2'], outputs=['y1'])
        self.cycle.promotes('d2', inputs=['z', 'y1'], outputs=['y2'])
        self.promotes('cycle', any=['*'])

        self.promotes('obj_cmp', any=['x', 'z', 'y1', 'y2', 'obj'])
        self.promotes('con_cmp1', any=['*'])
        self.promotes('con_cmp2', any=['con2', 'y2'])


prob = om.Problem()
prob.model = SellarMDA()

prob.setup()

prob['x'] = 2.
prob['z'] = [-1., -1.]

prob.run_model()