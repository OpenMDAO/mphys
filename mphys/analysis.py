""" this file holder the bases classes used by mphys"""


from openmdao.core.system import System
from openmdao.api import Group
from itertools import product, chain
from pprint import pprint
from .base_classes import SolverObjectBasedSystem, XferObject
from .mphys_meld import MELD_disp_xfer, MELD_load_xfer


# use parallele analysis for 
class Analysis(Group, SolverObjectBasedSystem):
    def initialize(self):
        # super().initialize(self)
        self.solvers_init = False
        self.options.declare('share_solver_objects', default=True)

        self.solver_objects = {}

    def init_solver_objects(self, comm):

        # create the systems and init the solver objects
        # TODO change this to use get/set methods?
        set_static_subs = set(self._subsystems_allprocs)
        set_subs = set(self._static_subsystems_allprocs)
        print( 'subs')
        for sub in set_static_subs | set_subs:
            print(sub.name)

        if self.options['share_solver_objects']:

            print(self.solver_objects)
            for sub in set_static_subs | set_subs:
                print('init ', sub.name)

                if isinstance(sub, SolverObjectBasedSystem):
                    for obj_key in sub.solver_objects:
                        if  obj_key in self.solver_objects and self.solver_objects[obj_key] != None:
                            print('set ', obj_key)
                            sub.set_solver_objects({obj_key: self.solver_objects[obj_key]})

                    if not sub.solvers_init:
                        sub.init_solver_objects(comm)

                        # same any new solver objects
                        self.solver_objects.update(sub.get_solver_objects())

        else:
            for sub in set_static_subs | set_subs:
                if isinstance(sub, SolverObjectBasedSystem):
                    if not sub.solvers_init:
                        sub.init_solver_objects(comm)

                    for obj_key in sub.solver_objects:
                        self.solver_objects[sub.name + '.' + obj_key] = sub.solver_objects[obj_key]

        self.solvers_init = True


    def setup(self):
        
        # create the solver objects if needed
        if not self.solvers_init:
            self.init_solver_objects(self.comm)
        
    def set_solver_objects(self, solver_objs):
        # recursively set the solver objects on all the sub systems
        super().set_solver_objects(solver_objs)


        set_static_subs = set(self._subsystems_allprocs)
        set_subs = set(self._static_subsystems_allprocs)
        
        print(solver_objs)
        for sub in set_static_subs | set_subs:
            if isinstance(sub, SolverObjectBasedSystem):
                for obj_key in solver_objs:
                    if obj_key in sub.solver_objects:
                        print('set ', sub.name)
                        sub.set_solver_objects({obj_key: solver_objs[obj_key]})



    def add_subsystem(self, *args, **kwargs):
        super().add_subsystem(*args, **kwargs)
        self.solver_objects.update(args[1].solver_objects)

    # def get_solver_objects(self):
    #     """ return a dictionary for the solvers for each subsystem """
    #     self.check_init_solvers()
        

    #     # solver_objects = {}
    #     # for analysis_name in analysis_data:          
    #     #     if isinstance(s, SolverObjectBasedSystem):
    #     #         solver_objects[analysis_name] = analysis_data['analysis'].get_solver_objects()
         
    #     return self.solver_objects

    # def set_solver_objects(self, given_objects):
    #     """ set a dictionary for the solvers for each subsystem """
        
    #     self.solver_objects.update(given_objects)
        # for sub in  self._subsystems_allprocs:
        #     print('set ', sub.name)

        #     if isinstance(sub, SolverObjectBasedSystem):
        #         for obj_key in given_objects:
        #             if obj_key in sub.solver_objects:
        #                 sub.solver_objects[obj_key] = self.solver_objects[obj_key]



        # solver_objects = {}
        # for analysis_name in analysis_data:          
        #     if isinstance(s, SolverObjectBasedSystem):
        #         analysis_data['analysis'].set_solver_objects(solver_objects[analysis_name])
         
   



class XferCoupledAnalysis(Analysis):
    """ this Analysis routine has support to share the size information needed by xfer classes """

 
    def configure(self):
        # add size information to any transfer components 
        # this is a temporary fix that will become unnecessary because of
        # PEOM 22

        print('hi')
        # for sys in self._subsystems_myproc:
        #     if not isinstance(sys, XferObject):
        #         sys._setup_var_data()

        self._setup_var_data()

        for sys in self.system_iter():
            if isinstance(sys, XferObject):


                if isinstance(sys, MELD_disp_xfer):
                    for var in ['x_s0', 'x_a0', 'u_s']:
                        
                        # get the data needed for this vairable from the other 
                        # inputs of the same name
                        var_abs = self._var_allprocs_prom2abs_list['input'][var]
                        var_meta = self._var_abs2meta[var_abs[0]]
                        sys.add_input(var, val=var_meta['value'], src_indices=var_meta['src_indices'])
                        self.promotes(sys.name, inputs=[var])

                    for var in ['u_a']:
                        
                        # u_a has the same shape as x_a0
                        var_abs = self._var_allprocs_prom2abs_list['input']['x_a0']
                        var_meta = self._var_abs2meta[var_abs[0]]
                        sys.add_output(var, shape=var_meta['shape'])                  
                        self.promotes(sys.name, outputs=[var])


        # inputs


                if isinstance(sys, MELD_load_xfer):
                    for var in ['x_s0', 'x_a0', 'u_s']:
                        
                        # get the data needed for this vairable from the other 
                        # inputs of the same name
                        var_abs = self._var_allprocs_prom2abs_list['input'][var]
                        var_meta = self._var_abs2meta[var_abs[0]]
                        sys.add_input(var, val=var_meta['value'], src_indices=var_meta['src_indices'])
                        self.promotes(sys.name, inputs=[var])
                    
                    for var in ['f_a']:
                        
                        # u_a has the same shape as x_a0
                        var_abs = self._var_allprocs_prom2abs_list['input']['x_a0']
                        var_meta = self._var_abs2meta[var_abs[0]]
                        sys.add_input(var, shape=var_meta['shape'], src_indices=var_meta['src_indices'] )
                        self.promotes(sys.name, inputs=[var])


                    for var in ['f_s']:
                        
                        # get the data needed for this vairable from the other 
                        # inputs of the same name
                        var_abs = self._var_allprocs_prom2abs_list['input']['u_s']
                        var_meta = self._var_abs2meta[var_abs[0]]
                        sys.add_output(var, shape=var_meta['shape'])
                        self.promotes(sys.name, outputs=[var])

                        

        # self.add_input('x_s0', shape = 0, src_indices = [], desc='initial structural node coordinates') #np.arange(sx1, sx2, dtype=int)
        # self.add_input('x_a0', shape = 0, src_indices = [], desc='initial aerodynamic surface node coordinates') #np.arange(ax1, ax2, dtype=int)
        # self.add_input('u_s',  shape = 0, src_indices = [], desc='structural node displacements') #np.arange(su1, su2, dtype=int)


        #         # look at the inputs
        #         for var in sys._var_rel_names['input']:
        #             # if the var shape  zero
        #             # set the shape to the shape of the other promoted var
        #             if sys._var_rel2meta[var]['size'] == 0:
        #                 print(sys.name, var)
        #                 pprint(self._var_allprocs_prom2abs_list['input'])
        #                 import ipdb; ipdb.set_trace()

        #                 var_abs = self._var_allprocs_prom2abs_list['input'][var]
        #                 var_meta = self._var_abs2meta[var_abs[0]]




        #                 # set the size using meta info
        #                 for data in ['value', 'shape', 'size']:
        #                     sys._var_rel2meta[var][data] = var_meta[data]




        #         # look at the outputs
        #         for var in sys._var_rel_names['output']:

        #             if sys._var_rel2meta[var]['size'] == 0:
        #                 print(sys.name, var)
        #                 var_abs = self._var_allprocs_prom2abs_list['output'][var]
        #                 var_meta = self._var_abs2meta[var_abs[0]]

        #                 # set the size using meta info
        #                 for data in ['value', 'shape', 'size']:
        #                     sys._var_rel2meta[var][data] = var_meta[data]


