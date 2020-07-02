
import numpy as np
import pprint
import copy

from collections import OrderedDict
from baseclasses import AeroProblem

from adflow import ADFLOW
from idwarp import USMesh

from openmdao.api import Group, ImplicitComponent, ExplicitComponent

from adflow.om_utils import get_dvs_and_cons

from .base_classes import SolverObjectBasedSystem
from .analysis import Analysis
class AdflowMesh(ExplicitComponent, SolverObjectBasedSystem):
    """
    Component to get the partitioned initial surface mesh coordinates

    """
    def initialize(self):
        self.options.declare('solver_options')

        self.options['distributed'] = True


        self.solver_objects = {'Adflow':None}
        
        # set the init flag to false
        self.solvers_init = False


    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver
            self.solvers_init = True

    def setup(self):

        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        self.x_a0 = self.solver_objects['Adflow'].mesh.getSurfaceCoordinates().flatten(order='C')

        coord_size = self.x_a0.size

        self.add_output('x_a0', shape=coord_size, desc='initial aerodynamic surface node coordinates')

    def mphys_add_coordinate_input(self):
        local_size = self.x_a0.size
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a0_points',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')

        # return the promoted name and coordinates
        return 'x_a0_points', self.x_a0

    def compute(self,inputs,outputs):
        if 'x_a0_points' in inputs:
            outputs['x_a0'] = inputs['x_a0_points']
        else:
            outputs['x_a0'] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'x_a0_points' in d_inputs:
                d_outputs['x_a0'] += d_inputs['x_a0_points']
        elif mode == 'rev':
            if 'x_a0_points' in d_inputs:
                d_inputs['x_a0_points'] += d_outputs['x_a0']

class Geo_Disp(ExplicitComponent, SolverObjectBasedSystem):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """


    def initialize(self):
        self.options.declare('solver_options')

        self.options['distributed'] = True


        self.solver_objects = {'Adflow':None}
        
        # set the init flag to false
        self.solvers_init = False


    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver
            self.solvers_init = True

    def setup(self):

        if not self.solvers_init:
            self.init_solver_objects(self.comm)


        aero_nnodes = self.solver_objects['Adflow'].getSurfaceCoordinates().size //3
        local_size = aero_nnodes * 3
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        #TODO set the val of the input x_a0
        self.add_input('x_a0',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')
        self.add_input('u_a', shape=local_size,val=np.zeros(local_size),src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface displacements')

        self.add_output('x_a',shape=local_size,desc='deformed aerodynamic surface')

    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_outputs['x_a'] += d_inputs['x_a0']
                if 'u_a' in d_inputs:
                    d_outputs['x_a'] += d_inputs['u_a']
        if mode == 'rev':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_inputs['x_a0'] += d_outputs['x_a']
                if 'u_a' in d_inputs:
                    d_inputs['u_a']  += d_outputs['x_a']

class AdflowWarper(ExplicitComponent, SolverObjectBasedSystem):
    """
    OpenMDAO component that wraps the warping.

    """

    def initialize(self):
        self.options.declare('solver_options')

        self.options['distributed'] = True


        self.solver_objects = {'Adflow':None}
        
        # set the init flag to false
        self.solvers_init = False


    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver
        
        self.solvers_init = True

    def setup(self):

        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        self.solver = self.solver_objects['Adflow']
        # self.add_output('foo', val=1.0)
        solver = self.solver

        # self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # state inputs and outputs
        local_coords = solver.getSurfaceCoordinates()
        local_coord_size = local_coords.size
        local_volume_coord_size = solver.mesh.getSolverGrid().size

        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a', src_indices=np.arange(n1,n2,dtype=int),shape=local_coord_size)

        self.add_output('x_g', shape=local_volume_coord_size)

        #self.declare_partials(of='x_g', wrt='x_s')

    def compute(self, inputs, outputs):

        solver = self.solver
        x_a = inputs['x_a'].reshape((-1,3))
        solver.setSurfaceCoordinates(x_a)
        solver.updateGeometryInfo()
        outputs['x_g'] = solver.mesh.getSolverGrid()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        solver = self.solver

        if mode == 'fwd':
            if 'x_g' in d_outputs:
                if 'x_a' in d_inputs:
                    dxS = d_inputs['x_a']
                    dxV = self.solver.mesh.warpDerivFwd(dxS)
                    d_outputs['x_g'] += dxV

        elif mode == 'rev':
            if 'x_g' in d_outputs:
                if 'x_a' in d_inputs:
                    dxV = d_outputs['x_g']
                    self.solver.mesh.warpDeriv(dxV)
                    dxS = self.solver.mesh.getdXs()
                    d_inputs['x_a'] += dxS.flatten()

class AdflowSolver(ImplicitComponent, SolverObjectBasedSystem):
    """
    OpenMDAO component that wraps the Adflow flow solver

    """

    def initialize(self):
        self.options.declare('solver_options')
        self.options.declare('aero_problem')
        #self.options.declare('use_OM_KSP', default=False, types=bool,
        #    desc="uses OpenMDAO's PestcKSP linear solver with Adflow's preconditioner to solve the adjoint.")

        self.options['distributed'] = True


        self.solver_objects = {'Adflow':None}
        
        # set the init flag to false
        self.solvers_init = False

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True

    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver

        self.solvers_init = True

    def setup(self):

        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        self.solver =self.solver_objects['Adflow']
        solver = self.solver

        # state inputs and outputs
        local_state_size = solver.getStateSize()
        local_coord_size = solver.mesh.getSolverGrid().size

        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_g', src_indices=np.arange(n1,n2,dtype=int),shape=local_coord_size)

        self.add_output('q', shape=local_state_size)


        self.set_ap(self.options['aero_problem'])

        #self.declare_partials(of='q', wrt='*')

    def _set_ap(self, inputs):
        tmp = {}
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            tmp[name] = inputs[name]

        self.ap.setDesignVars(tmp)

    def set_ap(self, ap):
        # this is the external function to set the ap to this component
        self.ap = copy.copy(ap)

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # parameter inputs
        if self.comm.rank == 0:
            print('adding ap var inputs')
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            self.add_input(name, shape=size, units=kwargs['units'])
            if self.comm.rank == 0:
                print(name)

    def _set_states(self, outputs):
        self.solver.setStates(outputs['q'])


    def apply_nonlinear(self, inputs, outputs, residuals):

        solver = self.solver

        self._set_states(outputs)
        self._set_ap(inputs)

        ap = self.ap

        # Set the warped mesh
        #solver.mesh.setSolverGrid(inputs['x_g'])
        # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now

        # flow residuals
        residuals['q'] = solver.getResidual(ap)


    def solve_nonlinear(self, inputs, outputs):

        solver = self.solver
        ap = self.ap

        if self._do_solve:

            # Set the warped mesh
            #solver.mesh.setSolverGrid(inputs['x_g'])
            # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now

            self._set_ap(inputs)
            ap.solveFailed = False # might need to clear this out?
            ap.fatalFail = False

            solver(ap)

        outputs['q'] = solver.getStates()


    def linearize(self, inputs, outputs, residuals):

        self.solver._setupAdjoint()

        self._set_ap(inputs)
        self._set_states(outputs)


    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):

        solver = self.solver
        ap = self.ap

        if mode == 'fwd':
            if 'q' in d_residuals:
                xDvDot = {}
                for var_name in d_inputs:
                    xDvDot[var_name] = d_inputs[var_name]
                if 'x_g' in d_inputs:
                    xVDot = d_inputs['x_g']
                else:
                    xVDot = None
                if 'q' in d_outputs:
                    wDot = d_outputs['q']
                else:
                    wDot = None

                dwdot = solver.computeJacobianVectorProductFwd(xDvDot=xDvDot,
                                                               xVDot=xVDot,
                                                               wDot=wDot,
                                                               residualDeriv=True)
                d_residuals['q'] += dwdot

        elif mode == 'rev':
            if 'q' in d_residuals:
                resBar = d_residuals['q']

                wBar, xVBar, xDVBar = solver.computeJacobianVectorProductBwd(
                    resBar=resBar,
                    wDeriv=True, xVDeriv=True, xDvDeriv=False, xDvDerivAero=True)

                if 'q' in d_outputs:
                    d_outputs['q'] += wBar

                if 'x_g' in d_inputs:
                    d_inputs['x_g'] += xVBar

                for dv_name, dv_bar in xDVBar.items():
                    if dv_name in d_inputs:
                        d_inputs[dv_name] += dv_bar.flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        solver = self.solver
        ap = self.ap
        if mode == 'fwd':
            d_outputs['q'] = solver.solveDirectForRHS(d_residuals['q'])
        elif mode == 'rev':
            #d_residuals['q'] = solver.solveAdjointForRHS(d_outputs['q'])
            solver.adflow.adjointapi.solveadjoint(d_outputs['q'], d_residuals['q'], True)

        return True, 0, 0

class AdflowForces(ExplicitComponent, SolverObjectBasedSystem):
    """
    OpenMDAO component that wraps force integration

    """


    def initialize(self):
        self.options.declare('solver_options')
        self.options.declare('aero_problem')
        #self.options.declare('use_OM_KSP', default=False, types=bool,
        #    desc="uses OpenMDAO's PestcKSP linear solver with Adflow's preconditioner to solve the adjoint.")

        self.options['distributed'] = True


        self.solver_objects = {'Adflow':None}
        
        # set the init flag to false
        self.solvers_init = False



    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver
        self.solvers_init = True

    def setup(self):

        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        self.solver =self.solver_objects['Adflow']

        solver = self.solver

        local_state_size = solver.getStateSize()
        local_coord_size = solver.mesh.getSolverGrid().size
        s_list = self.comm.allgather(local_state_size)
        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_g', src_indices=np.arange(n1,n2,dtype=int), shape=local_coord_size)
        self.add_input('q', src_indices=np.arange(s1,s2,dtype=int), shape=local_state_size)

        local_surface_coord_size = solver.mesh.getSurfaceCoordinates().size
        self.add_output('f_a', shape=local_surface_coord_size)


        self.set_ap(self.options['aero_problem'])

        # self.declare_partials(of='f_a', wrt='*')

    def _set_ap(self, inputs):
        tmp = {}
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            tmp[name] = inputs[name]

        self.ap.setDesignVars(tmp)

    def set_ap(self, ap):
        # this is the external function to set the ap to this component
        self.ap = ap

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # parameter inputs
        if self.comm.rank == 0:
            print('adding ap var inputs')
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            self.add_input(name, shape=size, units=kwargs['units'])
            if self.comm.rank == 0:
                print(name)

    def _set_states(self, inputs):
        self.solver.setStates(inputs['q'])

    def compute(self, inputs, outputs):

        solver = self.solver
        ap = self.ap

        self._set_ap(inputs)

        # Set the warped mesh
        #solver.mesh.setSolverGrid(inputs['x_g'])
        # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now
        self._set_states(inputs)

        outputs['f_a'] = solver.getForces().flatten(order='C')

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        solver = self.solver
        ap = self.ap

        if mode == 'fwd':
            if 'f_a' in d_outputs:
                xDvDot = {}
                for var_name in d_inputs:
                    xDvDot[var_name] = d_inputs[var_name]
                if 'q' in d_inputs:
                    wDot = d_inputs['q']
                else:
                    wDot = None
                if 'x_g' in d_inputs:
                    xVDot = d_inputs['x_g']
                else:
                    xVDot = None
                if not(xVDot is None and wDot is None):
                    dfdot = solver.computeJacobianVectorProductFwd(xDvDot=xDvDot,
                                                                   xVDot=xVDot,
                                                                   wDot=wDot,
                                                                   fDeriv=True)
                    d_outputs['f_a'] += dfdot.flatten()

        elif mode == 'rev':
            if 'f_a' in d_outputs:
                fBar = d_outputs['f_a']

                wBar, xVBar, xDVBar = solver.computeJacobianVectorProductBwd(
                    fBar=fBar,
                    wDeriv=True, xVDeriv=True, xDvDeriv=False, xDvDerivAero=True)

                if 'x_g' in d_inputs:
                    d_inputs['x_g'] += xVBar
                if 'q' in d_inputs:
                    d_inputs['q'] += wBar

                for dv_name, dv_bar in xDVBar.items():
                    if dv_name in d_inputs:
                        d_inputs[dv_name] += dv_bar.flatten()

FUNCS_UNITS={
    'mdot': 'kg/s',
    'mavgptot': 'Pa',
    'mavgps': 'Pa',
    'mavgttot': 'degK',
    'mavgvx':'m/s',
    'mavgvy':'m/s',
    'mavgvz':'m/s',
    'drag': 'N',
    'lift': 'N',
    'dragpressure': 'N',
    'dragviscous': 'N',
    'dragmomentum': 'N',
    'fx': 'N',
    'fy': 'N',
    'fz': 'N',
    'forcexpressure': 'N',
    'forceypressure': 'N',
    'forcezpressure': 'N',
    'forcexviscous': 'N',
    'forceyviscous': 'N',
    'forcezviscous': 'N',
    'forcexmomentum': 'N',
    'forceymomentum': 'N',
    'forcezmomentum': 'N',
    'flowpower': 'W',
    'area':'m**2',
}

class AdflowFunctions(ExplicitComponent, SolverObjectBasedSystem):

    def initialize(self):
        self.options.declare('solver_options')
        self.options.declare('aero_problem')
        #self.options.declare('use_OM_KSP', default=False, types=bool,
        #    desc="uses OpenMDAO's PestcKSP linear solver with Adflow's preconditioner to solve the adjoint.")



        self.solver_objects = {'Adflow':None}
        
        # set the init flag to false
        self.solvers_init = False

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True


    def init_solver_objects(self, comm):
        options = self.options['solver_options']

        #TODO add this code to an adflow component base class
        if self.solver_objects['Adflow'] == None:
            CFDSolver =  ADFLOW(options=self.options['solver_options'], comm=comm)
            
            # TODO there should be a sperate set of mesh options passed to USMesh
            # TODO the user should be able to choose the kind of mesh
            mesh = USMesh(options=self.options['solver_options'])
            CFDSolver.setMesh(mesh)
            self.solver_objects['Adflow'] = CFDSolver
            self.solvers_init = True

    def setup(self):

        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        self.solver =self.solver_objects['Adflow']
        solver = self.solver
        #self.set_check_partial_options(wrt='*',directional=True)

        local_state_size = solver.getStateSize()
        local_coord_size = solver.mesh.getSolverGrid().size
        s_list = self.comm.allgather(local_state_size)
        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_g', src_indices=np.arange(n1,n2,dtype=int), shape=local_coord_size)
        self.add_input('q', src_indices=np.arange(s1,s2,dtype=int), shape=local_state_size)

        self.set_ap(self.options['aero_problem'])
            #self.declare_partials(of=f_name, wrt='*')

    def _set_ap(self, inputs):
        tmp = {}
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            tmp[name] = inputs[name][0]

        self.ap.setDesignVars(tmp)
        #self.options['solver'].setAeroProblem(self.options['ap'])

    def set_ap(self, ap):
        # this is the external function to set the ap to this component
        self.ap = copy.copy(ap)

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # parameter inputs
        if self.comm.rank == 0:
            print('adding ap var inputs')
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            self.add_input(name, shape=size, units=kwargs['units'])
            if self.comm.rank == 0:
                print(name)

        for f_name in self.ap.evalFuncs:

            if self.comm.rank == 0:
                print("adding adflow func as output: {}".format(f_name))
            self.add_output(f_name, shape=1)
            #self.add_output(f_name, shape=1, units=units)

    def _set_states(self, inputs):
        self.solver.setStates(inputs['q'])

    def _get_func_name(self, name):
        return '%s_%s' % (self.ap.name, name.lower())

    def compute(self, inputs, outputs):
        solver = self.solver
        ap = self.ap
        #print('funcs compute')
        #actually setting things here triggers some kind of reset, so we only do it if you're actually solving
        if self._do_solve:
            self._set_ap(inputs)
            # Set the warped mesh
            #solver.mesh.setSolverGrid(inputs['x_g'])
            # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now
            self._set_states(inputs)

        funcs = {}

        eval_funcs = [f_name for f_name in self.ap.evalFuncs]
        solver.evalFunctions(ap, funcs, eval_funcs)

        for name in self.ap.evalFuncs:
            f_name = self._get_func_name(name)
            if f_name in funcs:
                outputs[name.lower()] = funcs[f_name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        solver = self.solver
        ap = self.ap

        if mode == 'fwd':
            xDvDot = {}
            for key in ap.DVs:
                if key in d_inputs:
                    mach_name = key.split('_')[0]
                    xDvDot[mach_name] = d_inputs[key]

            if 'q' in d_inputs:
                wDot = d_inputs['q']
            else:
                wDot = None

            if 'x_g' in d_inputs:
                xVDot = d_inputs['x_g']
            else:
                xVDot = None

            funcsdot = solver.computeJacobianVectorProductFwd(
                xDvDot=xDvDot,
                xVDot=xVDot,
                wDot=wDot,
                funcDeriv=True)

            for name in funcsdot:
                func_name = name.lower()
                if name in d_outputs:
                    d_outputs[name] += funcsdot[func_name]

        elif mode == 'rev':
            funcsBar = {}

            for name  in self.ap.evalFuncs:
                func_name = name.lower()

                # we have to check for 0 here, so we don't include any unnecessary variables in funcsBar
                # becasue it causes Adflow to do extra work internally even if you give it extra variables, even if the seed is 0
                if func_name in d_outputs and d_outputs[func_name] != 0.:
                    funcsBar[func_name] = d_outputs[func_name][0]
                    # this stuff is fixed now. no need to divide
                    # funcsBar[func_name] = d_outputs[func_name][0] / self.comm.size
                    # print(self.comm.rank, func_name, funcsBar[func_name])

            #print(funcsBar, flush=True)

            d_input_vars = list(d_inputs.keys())
            n_input_vars = len(d_input_vars)

            wBar = None
            xVBar = None
            xDVBar = None

            wBar, xVBar, xDVBar = solver.computeJacobianVectorProductBwd(
                funcsBar=funcsBar,
                wDeriv=True, xVDeriv=True, xDvDeriv=False, xDvDerivAero=True)
            if 'q' in d_inputs:
                d_inputs['q'] += wBar
            if 'x_g' in d_inputs:
                d_inputs['x_g'] += xVBar

            for dv_name, dv_bar in xDVBar.items():
                if dv_name in d_inputs:
                    d_inputs[dv_name] += dv_bar.flatten()



class AdflowGroup(Analysis):

    def initialize(self):
        super().initialize()
        self.options.declare('aero_problem')
        self.options.declare('solver_options')
        self.options.declare('group_options')

        # default values which are updated later
        self.group_options = {
            'mesh': True,
            'geo_disp': False,
            'deformer': True,
            'solver': True,
            'funcs': True,
            'forces': False,
        }
        self.group_components = OrderedDict({
            'mesh': AdflowMesh,
            'geo_disp': Geo_Disp,
            'deformer': AdflowWarper,
            'solver': AdflowSolver,
            'funcs': AdflowFunctions,
            'forces': AdflowForces,
        })

        # self.solver_objects = {'Adflow':None}


        self.solvers_init = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.group_options.update(self.options['group_options'])

        # if you wanted to check that the user gave a valid combination of components (solver, mesh, ect)
        # you could do that here, but they will be shown on the n2 


        print("=========")
        for comp in self.group_components:
            if self.group_options[comp]:
                print(comp)
                if comp in ['mesh', 'geo_disp', 'deformer']:
                    self.add_subsystem(comp, self.group_components[comp](solver_options=self.options['solver_options']),
                                        promotes=['*']) # we can connect things implicitly through promotes
                                                        # because we already know the inputs and outputs of each
                                                        # components 
                
                else:
                    self.add_subsystem(comp, self.group_components[comp](solver_options=self.options['solver_options'],
                                                                         aero_problem=self.options['aero_problem']),
                                                    promotes=['*']) # we can connect things implicitly through promotes
                                                                    # because we already know the inputs and outputs of each
                                                                    # components 
        

    def setup(self):
        # issue conditional connections
        if self.group_options['mesh'] and self.group_options['deformer'] and not self.group_options['geo_disp']:
            self.connect('x_a0', 'x_a')

        if not self.solvers_init:
            self.init_solver_objects(self.comm)

        


