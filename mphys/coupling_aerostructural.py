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
