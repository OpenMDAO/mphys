""" this file holder the bases classes used by mphys"""

from abc import ABCMeta, abstractmethod

from openmdao.core.system import System
from openmdao.api import Group
from itertools import product, chain
from pprint import pprint


class SolverObjectBasedSystem(metaclass=ABCMeta):
    """
        the base class for analysis in MPhys that require a solver object.
    """

    # the following mush be set in the initialize method
    # def initialize(self):

        #initialize the solver objects to None
        # self.solver_objects = { 'MeshName':None, 
        #                     'SolverName':None,
        #                     'EctName':None}
        
        # set the init flag to false
        # self.solvers_init = False

    
    @abstractmethod
    def init_solver_objects(self, comm):
        """ creates solver/transfer scheme/etc using provided comm which
        entails the allocation of memory for any computation"""
        
        mesh = mymodule.MeshObj(self.options['mesh_file'])
        solver = mymodule.SolverObj(self.options['solver_options'])
        
        self.solver_objects = { 'MeshName':mesh, 
                                'SolverName':solver,
                                'EctName':ect}
        
        # set the init flag to true!
        self.solvers_init = True



    def get_solver_objects(self):
        self.check_init_solvers()
        return self.solver_objects

    def set_solver_objects(self, solver_objects):
        self.solver_objects.update(solver_objects)

        # if all of the dictionary values for the solver_objects dict are not none
        if all(self.solver_objects.values()):
            self.solvers_init  = True


    def check_init_solvers(self):
        if not self.solvers_init:
            import ipdb; ipdb.set_trace()
            raise RuntimeError('Solver used before it was initialized or set')


class XferObject(SolverObjectBasedSystem):
    pass


   