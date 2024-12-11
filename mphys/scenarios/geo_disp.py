import openmdao.api as om
from mphys import MPhysVariables

class GeoDisp(om.ExplicitComponent):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options.declare('number_of_nodes')

    def setup(self):
        nnodes = self.options['number_of_nodes']
        local_size = nnodes * 3

        self.x_aero0_name = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
        self.u_aero_name = MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS
        self.x_aero_name = MPhysVariables.Aerodynamics.Surface.COORDINATES

        self.add_input(self.x_aero0_name,
                       shape_by_conn=True,
                       distributed=True,
                       desc='aerodynamic surface with geom changes',
                       tags=['mphys_coordinates'])
        self.add_input(self.u_aero_name,
                       shape_by_conn=True,
                       distributed=True,
                       desc='aerodynamic surface displacements',
                       tags=['mphys_coupling'])

        self.add_output(self.x_aero_name,
                        shape=local_size,
                        distributed=True,
                        desc='deformed aerodynamic surface',
                        tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        outputs[self.x_aero_name] = inputs[self.x_aero0_name] + inputs[self.u_aero_name]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if self.x_aero_name in d_outputs:
                if self.x_aero0_name in d_inputs:
                    d_outputs[self.x_aero_name] += d_inputs[self.x_aero0_name]
                if self.u_aero_name in d_inputs:
                    d_outputs[self.x_aero_name] += d_inputs[self.u_aero_name]
        if mode == 'rev':
            if self.x_aero_name in d_outputs:
                if self.x_aero0_name in d_inputs:
                    d_inputs[self.x_aero0_name] += d_outputs[self.x_aero_name]
                if self.u_aero_name in d_inputs:
                    d_inputs[self.u_aero_name]  += d_outputs[self.x_aero_name]
