import numpy as np
import openmdao.api as om


class GeometrySum(om.ExplicitComponent):
    def setup(self):
        self.add_input('ctrl_pt_delta', shape_by_conn=True)
        self.add_output('total_movement')
        self.declare_partials(['*'],['*'],method='fd')

    def compute(self, inputs, outputs):
        total = 0.0
        for i in range(inputs['ctrl_pt_delta'].shape[0]):
            for j in range(inputs['ctrl_pt_delta'].shape[1]):
                total += np.sqrt(inputs['ctrl_pt_delta'][i,j,0]**2.0+inputs['ctrl_pt_delta'][i,j,1]**2.0)
        outputs['total_movement'] = total
