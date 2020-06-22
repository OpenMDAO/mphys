""" this file holder the bases classes used by mphys"""

from abc import ABCMeta, abstractmethod

from openmdao.core.system import System
from openmdao.api import Group

class SolverObjectBasedSystem(metaclass=ABCMeta):
    """
        the base class for analysis in MPhys that require a solver object.
    """
    
    @abstractmethod
    def init_solver_objects(self, comm):
        """ contracts solver/transfer scheme/etc using provided comm which
        entails the allocation of memory for any computation"""
        self.solver = None
        
        self.solver_init = True



    def get_solver_objects(self):
        self.check_init_solver()
        return self.solver_object

    def set_solver_objects(self, solver_object):
        self.solver_object = solver_object

    def check_init_solvers(self):
        if not self.solver_init:
            raise RuntimeError('Solver used before it was initialized or set')


class CoupledAnalysis(om.Group, SolverObjectBasedSystem):
    def initialize(self):
        self.options.declare('solver_options')
        self.options.declare('group_options')