import numpy as np
import openmdao.api as om


class SpeedIndexComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('flutter_q')
        self.add_input('pitch_frequency')
        self.add_output('vf')

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        b = 0.5
        mu =  60.
        rho = 1.225
        vel = np.sqrt(2.0 * inputs['flutter_q']/ rho)
        outputs['vf'] = vel / ( b * inputs['pitch_frequency'] * np.sqrt(mu))
