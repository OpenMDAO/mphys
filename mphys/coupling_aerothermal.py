import openmdao.api as om
from .coupling_group import CouplingGroup
from .geo_disp import GeoDisp


class CouplingAeroThermal(CouplingGroup):
    """
    The standard aerothermal coupling problem.
    """

    def initialize(self):
        self.options.declare('aero_builder', recordable=False)
        self.options.declare('thermal_builder', recordable=False)
        self.options.declare('thermalxfer_builder', recordable=False)
        self.options.declare("scenario_name", recordable=True, default=None)

    def setup(self):
        aero_builder = self.options['aero_builder']
        
        thermal_builder = self.options['thermal_builder']
        
        thermalxfer_builder = self.options['thermalxfer_builder']
        
        scenario_name = self.options['scenario_name']

        heat_xfer, temp_xfer = thermalxfer_builder.get_coupling_group_subsystem(scenario_name)
        
        aero = aero_builder.get_coupling_group_subsystem(scenario_name)
        thermal = thermal_builder.get_coupling_group_subsystem(scenario_name)

        # geo_disp = GeoDisp(number_of_nodes=aero_builder.get_number_of_nodes())

        self.mphys_add_subsystem('thermal', thermal)
        self.mphys_add_subsystem('temp_xfer', temp_xfer)
        self.mphys_add_subsystem('aero', aero)
        self.mphys_add_subsystem('heat_xfer', heat_xfer)

        # add the defualt iterative solvers
        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=25, iprint=2,
                                                    atol=1e-8, rtol=1e-8,
                                                    use_aitken=True)
        self.linear_solver = om.LinearBlockGS(maxiter=25, iprint=2,
                                              atol=1e-8, rtol=1e-8,
                                              use_aitken=True)
