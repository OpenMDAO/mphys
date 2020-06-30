""" this file holder the bases classes used by mphys"""

from abc import ABCMeta, abstractmethod

from openmdao.core.system import System
from openmdao.api import Group
from itertools import product, chain


class SolverObjectBasedSystem(metaclass=ABCMeta):
    """
        the base class for analysis in MPhys that require a solver object.
    """
    
    @abstractmethod
    def init_solver_objects(self, comm):
        """ creates solver/transfer scheme/etc using provided comm which
        entails the allocation of memory for any computation"""
        self.solver_objects = None
        
        self.solvers_init = True



    def get_solver_objects(self):
        self.check_init_solvers()
        return self.solver_objects

    def set_solver_objects(self, solver_objects):
        self.solver_objects = solver_objects
        self.solvers_init  = True

    def check_init_solvers(self):
        if not self.solvers_init:
            raise RuntimeError('Solver used before it was initialized or set')


class XferObject(SolverObjectBasedSystem):
    pass

# class XferAnalysis(CoupledAnalysis):
#     """ Analysis where the solver objects are shared by all the xfer objects"""

# # maybe add this to analysis as an option?


class XferCoupledAnalysis(Group, SolverObjectBasedSystem):
    """ this Analysis routine has support to share the size information needed by xfer classes """

    def initialize(self):
        # define the inputs we need
        self.solvers_init  = False


    def setup(self):
        
        # create the solver objects if needed
        if not self.solvers_init:
            self.init_solver_objects(self.comm)
        

        # # add the substems
        # analysis_data = self.options['analyses'] 
        # for analysis_name in analysis_data:
        #     ana_data = analysis_data[analysis_name]

        #     if 'subsystem_options' in ana_data:
        #         ana = self.add_subsystem(analysis_name, ana_data['analysis'], **ana_data['subsystem_options']  ) 
        #     else:
        #         ana = self.add_subsystem(analysis_name, ana_data['analysis'])
 
    def configure(self):
        # add size information to any transfer components 
        # this is a temporary fix that will hopefull become unnecessary because of
        # PEOM 22

        print('hi')
        self._setup_var_data()

        for sys in self.system_iter():
            if isinstance(sys, C1):

                # look at the inputs
                for var in sys._var_rel_names['input']:
                    # if the var shape  zero
                    # set the shape to the shape of the other promoted var
                    if sys._var_rel2meta[var]['size'] == 0:
                        print(sys.name, var)
                        var_abs = self._var_allprocs_prom2abs_list['output'][var]
                        var_meta = self._var_abs2meta[var_abs[0]]




                        # set the size using meta info
                        for data in ['value', 'shape', 'size']:
                            sys._var_rel2meta[var][data] = var_meta[data]




                # look at the outputs
                for var in sys._var_rel_names['output']:

                    if sys._var_rel2meta[var]['size'] == 0:
                        print(sys.name, var)
                        var_abs = self._var_allprocs_prom2abs_list['input'][var]
                        var_meta = self._var_abs2meta[var_abs[0]]

                        # set the size using meta info
                        for data in ['value', 'shape', 'size']:
                            sys._var_rel2meta[var][data] = var_meta[data]



    def init_solver_objects(self, comm):

        # create the systems and init the solver objects

        for sub in  self._subsystems_allprocs:
            if isinstance(sub, SolverObjectBasedSystem):
                sub.init_solver_objects(comm)
                print('init ', sub.name)
        
        # for analysis_name in analysis_data:          
        #     ana = analysis_data[analysis_name]
        #         ana['analysis'].init_solver_objects(comm)
        #     else:
        #         #each analysis will must initialize its own solver objects later
        #         pass

        self.solver_init = True

    def get_solver_objects(self):
        """ return a dictionary for the solvers for each subsystem """
        self.check_init_solver()
        

        solver_objects = {}
        for analysis_name in analysis_data:          
            if isinstance(s, SolverObjectBasedSystem):
                solver_objects[analysis_name] = analysis_data['analysis'].get_solver_objects()
         
        return solver_objects

    def get_solver_objects(self):
        """ set a dictionary for the solvers for each subsystem """
        

        solver_objects = {}
        for analysis_name in analysis_data:          
            if isinstance(s, SolverObjectBasedSystem):
                analysis_data['analysis'].set_solver_objects(solver_objects[analysis_name])
         
   