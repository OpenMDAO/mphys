from collections import OrderedDict
import openmdao.api as om
from mphys.mphys_scenario import MPHYS_Scenario
from mphys.mphys_error import MPHYS_Error
from .base_classes import SolverObjectBasedSystem
import copy

class SerialMultipoint(om.Group):

    def initialize(self):
        # define the inputs we need
        self.options.declare('scenerio_analysis')
        self.options.declare('scenario_data')
        self.options.declare('share_solver_object', default=True)


    def setup(self):
        first_init = True

        scen_anys = self.options['scenerio_analysis']
        scen_data = self.options['scenario_data']

        # iterate through the scenario_data and create a scenarios and set data
        for scenario_name in scen_data:
            s_data = scen_data[scenario_name]
            if 'subsystem_options' in s_data:
                s = self.add_subsystem(scenario_name, scen_anys(**s_data['analysis_options']), **s_data['subsystem_options']  ) 
            else:
                s = self.add_subsystem(scenario_name, scen_anys(**s_data['analysis_options']) ) 
                print(scenario_name, s)
        
            if isinstance(s, SolverObjectBasedSystem) and  self.options['share_solver_object']:
                    
                    if first_init:
                        s.init_solver_objects(self.comm)
                        solver_object = s.get_solver_object()
                        first_init = False
                    else:
                        s.set_solver_objects(solver_object)
            else:
                #each analysis will must initialize its own solver objects 
                pass





class ParallelMultipoint(om.ParallelGroup):

    def initialize(self):
        # define the inputs we need
        self.options.declare('scenerio_analysis')
        self.options.declare('scenario_data')

    def setup(self):
        scen_anys = self.options['scenerio_analysis']
        scen_data = self.options['scenario_data']

        # iterate through the scenario_data and create a scenarios and set data
        for scenario_name in scen_data:
            s_data = scen_data[scenario_name]
            if 'subsystem_options' in s_data:
                s = self.add_subsystem(scenario_name, scen_anys(**s_data['analysis_options']), **s_data['subsystem_options']  ) 
            else:
                s = self.add_subsystem(scenario_name, scen_anys(**s_data['analysis_options']) ) 







 