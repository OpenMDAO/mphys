from .coupling_group import CouplingGroup
from .geo_disp import GeoDisp


class CouplingAeroStructural(CouplingGroup):
    """
    The standard aerostructural coupling problem.
    """

    def initialize(self):
        self.options.declare('aero_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('xfer_builder', recordable=False)

    def setup(self):
        aero_builder = self.options['aero_builder']
        struct_builder = self.options['struct_builder']
        xfer_builder = self.options['xfer_builder']

        disp_xfer, load_xfer = xfer_builder.get_coupling_group_subsystem()
        aero = aero_builder.get_coupling_group_subsystem()
        struct = struct_builder.get_coupling_group_subsystem()

        geo_disp = GeoDisp(number_of_nodes=aero_builder.get_number_of_nodes())

        self.mphys_add_subsystem('disp_xfer', disp_xfer)
        self.mphys_add_subsystem('geo_disp', geo_disp)
        self.mphys_add_subsystem('aero', aero)
        self.mphys_add_subsystem('load_xfer', load_xfer)
        self.mphys_add_subsystem('struct', struct)

    def configure(self):
        self.connect('disp_xfer.u_aero', 'geo_disp.u_aero')
        self.connect('geo_disp.x_aero', 'aero.x_aero')
        self.connect('aero.f_aero', 'load_xfer.f_aero')
        self.connect('load_xfer.f_struct', 'struct.f_struct')
        self.connect('struct.u_struct', 'disp_xfer.u_struct')

        # only nonlinear xfers have load_xfer.u_struct
        if self._mphys_variable_is_in_subsystem_inputs(self.load_xfer, 'u_struct'):
            self.connect('struct.u_struct', ['load_xfer.u_struct'])

        super().configure()
