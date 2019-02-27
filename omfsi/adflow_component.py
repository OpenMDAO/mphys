
import numpy as np 
import pprint

from baseclasses import AeroProblem

from adflow import ADFLOW

from openmdao.api import ImplicitComponent
from openmdao.core.analysis_error import AnalysisError

from .om_utils import get_dvs_and_cons

class AdflowMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates

    """
    def initialize(self):
        self.options.declare('ap', types=AeroProblem)
        self.options.declare('solver')
        self.options['distributed'] = True

    def setup(self):
        self.x_a0 = self.options.['solver'].getSurfaceCoordinates())
        coord_size = self.x_a0.size

        self.add_output('x_a0', shape=coord_size, desc='initial aerodynamic surface node coordinates')

    def compute(self,inputs,outputs):
        outputs['x_a0'] = self.x_a0

class AdflowSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the warping + flow solve

    """

    def initialize(self):
        self.options.declare('ap', types=AeroProblem)
        self.options.declare('solver')
        #self.options.declare('use_OM_KSP', default=False, types=bool, 
        #    desc="uses OpenMDAO's PestcKSP linear solver with ADflow's preconditioner to solve the adjoint.")

        self.distributed = True

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True

    def setup(self):
        solver = self.options['solver']
        ap = self.options['ap']

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

        # state inputs and outputs
        local_state_size = solver.getStateSize()
        local_coord_size = solver.getSurfaceCoordinates().size

        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a', src_indices=np.arange(n1,n2,dtype=int),shape=local_coord_size)

        self.add_output('q', shape=local_state_size)
        
        #self.declare_partials(of='q', wrt='*')

    def _set_ap(self, inputs):
        tmp = {}
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            tmp[name] = inputs[name]

        self.options['ap'].setDesignVars(tmp)

    def _set_states(self, outputs):
        self.options['solver'].setStates(outputs['q'])
        
    
    def apply_nonlinear(self, inputs, outputs, residuals):
        
        solver = self.options['solver']

        self._set_states(outputs)
        self._set_ap(inputs)
        
        ap = self.options['ap']

        # set displaced surface and warp the volume
        x_a = inputs['x_a'] # reshape?
        solver.setSurfaceCoordinates(x_a)
        solver.updateGeometryInfo()

        # flow residuals
        residuals['q'] = solver.getResidual(ap)

    
    def solve_nonlinear(self, inputs, outputs):

        solver = self.options['solver']
        ap = self.options['ap']

        if self._do_solve: 

            self._set_ap(inputs)
            ap.solveFailed = False # might need to clear this out?
            ap.fatalFail = False
        
            x_a = inputs['x_a'] # reshape?
            solver.setSurfaceCoordinates(x_a)
            solver.updateGeometryInfo()
            solver(ap)

            if ap.fatalFail:
                if self.comm.rank == 0:
                    print('###############################################################')
                    print('#Solve Fatal Fail. Analysis Error')
                    print('###############################################################')

                raise AnalysisError('ADFLOW Solver Fatal Fail')


            if ap.solveFailed: # the mesh was fine, but it didn't converge
                if self.comm.rank == 0:
                    print('###############################################################')
                    print('#Solve Failed, attempting a clean restart!')
                    print('###############################################################')

                ap.solveFailed = False
                ap.fatalFail = False
                solver.resetFlow(ap)
                solver(ap)

                if ap.solveFailed or ap.fatalFail: # we tried, but there was no saving it
                    print('###############################################################')
                    print('#Clean Restart failed. There is no saving this one!')
                    print('###############################################################')

                    raise AnalysisError('ADFLOW Solver Fatal Fail')

        outputs['q'] = solver.getStates()
        outputs['f_a'] = solver.getForces()

    
    def linearize(self, inputs, outputs, residuals):

        self.options['solver']._setupAdjoint()

        self._set_ap(inputs)
        self._set_states(outputs)

        #print('om_states linearize')
        
    
    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):

        solver = self.options['solver']
        ap = self.options['ap']

        if mode == 'fwd':
            if 'q' in d_residuals:
                xDvDot = {}
                for var_name in d_inputs:
                    xDvDot[var_name] = d_inputs[var_name]
                if 'x_a' in d_inputs:
                    xSdot = d_inputs['x_a']
                else:
                    xSdot = None
                if 'q' in d_outputs:
                    wDot = d_outputs['q']
                else:
                    wDot = None

                dwdot = solver.computeJacobianVectorProductFwd(xDvDot=xDvDot,
                                                               xSDot=xSDot,
                                                               wDot=wDot,
                                                               residualDeriv=True)
                d_residuals['q'] += dwdot

        elif mode == 'rev':
            if 'q' in d_residuals:
                resBar = d_residuals['q']

                wBar, xSBar, xDVBar = solver.computeJacobianVectorProductBwd(
                    resBar=resBar,
                    wDeriv=True, xSDeriv=True, xDvDeriv=True)

                if 'q' in d_outputs:
                    d_outputs['q'] += wBar

                if 'x_a' in d_inputs:
                    d_outputs['x_a'] += xSBar

                for dv_name, dv_bar in xDVBar.items():
                    if dv_name in d_inputs:
                        d_inputs[dv_name] += dv_bar.flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        solver = self.options['solver']
        ap = self.options['ap']
        if self.options['use_OM_KSP']:
                if mode == 'fwd':
                    d_outputs['q'] = solver.globalNKPreCon(d_residuals['q'], d_outputs['q'])
                elif mode == 'rev':
                    d_residuals['q'] = solver.globalAdjointPreCon(d_outputs['q'], d_residuals['q'])
        else:
            if mode == 'fwd':
                d_outputs['q'] = solver.solveDirectForRHS(d_residuals['q'])
            elif mode == 'rev':
                #d_residuals['q'] = solver.solveAdjointForRHS(d_outputs['q'])
                solver.adflow.adjointapi.solveadjoint(d_outputs['q'], d_residuals['q'], True)

        return True, 0, 0

class AdflowForces(ExplicitComponent):
    """
    OpenMDAO component that wraps force integration

    """

    def initialize(self):
        self.options.declare('ap', types=AeroProblem)
        self.options.declare('solver')
        #self.options.declare('use_OM_KSP', default=False, types=bool, 
        #    desc="uses OpenMDAO's PestcKSP linear solver with ADflow's preconditioner to solve the adjoint.")

        self.distributed = True

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True

    def setup(self):
        solver = self.options['solver']
        ap = self.options['ap']

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        if self.comm.rank == 0: 
            print('adding ap var inputs')
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            self.add_input(name, shape=size, units=kwargs['units'])
            if self.comm.rank == 0: 
                print(name)


        local_state_size = solver.getStateSize()
        local_coord_size = solver.getSurfaceCoordinates().size
        s_list = self.comm.allgather(local_state_size)
        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a', src_indices=np.arange(n1,n2,dtype=int), shape=local_coord_size)
        self.add_input('q', src_indices=np.arange(s1,s2,dtype=int), shape=local_state_size)

        self.add_output('f_a', shape=local_coord_size)
        
        #self.declare_partials(of='f_a', wrt='*')

    def _set_ap(self, inputs):
        tmp = {}
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            tmp[name] = inputs[name]

        self.options['ap'].setDesignVars(tmp)

    def _set_states(self, inputs):
        self.options['solver'].setStates(inputs['q'])
    
    def compute(self, inputs, outputs):

        solver = self.options['solver']
        ap = self.options['ap']

        self._set_ap(inputs)
    
        x_a = inputs['x_a'] # reshape
        solver.setSurfaceCoordinates(x_a)
        solver.updateGeometryInfo()
        self._set_states(inputs)

        outputs['f_a'] = solver.getForces()
    
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        solver = self.options['solver']
        ap = self.options['ap']

        if mode == 'fwd':
            if 'f_a' in d_outputs:
                if 'q' in d_inputs:
                    wDot = d_inputs['q']
                else:
                    wDot = None
                if 'q' in d_inputs:
                    wDot = d_inputs['q']
                else:
                    wDot = None
                if 'x_a' in d_inputs:
                    xSdot = d_inputs['x_a']
                else:
                    xSdot = None

                dfdot = solver.computeJacobianVectorProductFwd(xSDot=xSDot,
                                                               wDot=wDot,
                                                               fDeriv=True)
                d_outputs['f_a'] += dfdot

        elif mode == 'rev':
            if 'q' in d_outputs:
                fBar = d_outputs['q']

                wBar, xSBar, xDVBar = solver.computeJacobianVectorProductBwd(
                    fBar=fBar,
                    wDeriv=True, xSDeriv=True, xDvDeriv=True)

                if 'x_a' in d_inputs:
                    d_outputs['x_a'] += xSBar
                if 'q' in d_inputs:
                    d_outputs['q'] += wBar
