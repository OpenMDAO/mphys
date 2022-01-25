import openmdao.api as om
from .coupling_group import CouplingGroup
from .geo_disp import GeoDisp


class CouplingAeroStructural(CouplingGroup):
    """
    The standard aerostructural coupling problem.
    """

    def initialize(self):
        self.options.declare('aero_builder', recordable=False)
        self.options.declare('struct_builder', recordable=False)
        self.options.declare('ldxfer_builder', recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        aero_builder = self.options['aero_builder']
        struct_builder = self.options['struct_builder']
        ldxfer_builder = self.options['ldxfer_builder']
        scenario_name = self.options['scenario_name']

        disp_xfer, load_xfer = ldxfer_builder.get_coupling_group_subsystem(scenario_name)
        aero = aero_builder.get_coupling_group_subsystem(scenario_name)
        struct = struct_builder.get_coupling_group_subsystem(scenario_name)

        geo_disp = GeoDisp(number_of_nodes=aero_builder.get_number_of_nodes())

        self.mphys_add_subsystem('disp_xfer', disp_xfer)
        self.mphys_add_subsystem('geo_disp', geo_disp)
        self.mphys_add_subsystem('aero', aero)
        self.mphys_add_subsystem('load_xfer', load_xfer)
        self.mphys_add_subsystem('struct', struct)

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2,
                                                    atol=1e-8, rtol=1e-8,
                                                    use_aitken=True)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2,
                                              atol=1e-8, rtol=1e-8,
                                              use_aitken=True)
