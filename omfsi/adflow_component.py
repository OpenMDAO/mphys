
import numpy as np
import pprint

from baseclasses import AeroProblem

from adflow import ADFLOW
from pywarp import MBMesh

from openmdao.api import ImplicitComponent, ExplicitComponent
from openmdao.core.analysis_error import AnalysisError

from adflow.python.om_utils import get_dvs_and_cons

class AdflowMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates

    """
    def initialize(self):
        self.options.declare('ap', types=AeroProblem)
        self.options.declare('solver')
        self.options.declare('options')
        self.options['distributed'] = True

    def setup(self):
        self.mesh = MBMesh(comm=self.comm,options=self.options['options'])
        self.options['solver'].setMesh(self.mesh)

        self.x_a0 = self.mesh.getSurfaceCoordinates().flatten(order='C')

        coord_size = self.x_a0.size

        self.add_output('x_a0', shape=coord_size, desc='initial aerodynamic surface node coordinates')

    def compute(self,inputs,outputs):
        outputs['x_a0'] = self.x_a0

class AdflowWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.

    """

    def initialize(self):
        self.options.declare('ap', types=AeroProblem)
        self.options.declare('solver')
        #self.options.declare('use_OM_KSP', default=False, types=bool,
        #    desc="uses OpenMDAO's PestcKSP linear solver with ADflow's preconditioner to solve the adjoint.")

        self.distributed = True

    def setup(self):
        solver = self.options['solver']
        ap = self.options['ap']

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # state inputs and outputs
        local_coord_size = solver.getSurfaceCoordinates().size
        local_volume_coord_size = solver.mesh.getSolverGrid().size

        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a', src_indices=np.arange(n1,n2,dtype=int),shape=local_coord_size)

        self.add_output('x_g', shape=local_volume_coord_size)

        #self.declare_partials(of='x_g', wrt='x_s')

    def compute(self, inputs, outputs):

        solver = self.options['solver']

        x_a = inputs['x_a'].reshape((-1,3))
        solver.setSurfaceCoordinates(x_a)
        solver.updateGeometryInfo()
        outputs['x_g'] = solver.mesh.getSolverGrid()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        solver = self.options['solver']
        ap = self.options['ap']

        if mode == 'fwd':
            if 'x_g' in d_outputs:
                if 'x_a' in d_inputs:
                    dxS = d_inputs['x_a']
                    dxV = self.options['solver'].mesh.warpDerivFwd(dxS)
                    d_outputs['x_g'] += dxV

        elif mode == 'rev':
            if 'x_g' in d_outputs:
                if 'x_a' in d_inputs:
                    dxV = d_outputs['x_g']
                    self.options['solver'].mesh.warpDeriv(dxV)
                    dxS = self.options['solver'].mesh.getdXs()
                    d_inputs['x_a'] += dxS.flatten()

class AdflowSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the ADflow flow solver

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
        local_coord_size = solver.mesh.getSolverGrid().size

        n_list = self.comm.allgather(local_coord_size)
        irank  = self.comm.rank
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_g', src_indices=np.arange(n1,n2,dtype=int),shape=local_coord_size)

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

        # Set the warped mesh
        #solver.mesh.setSolverGrid(inputs['x_g'])
        # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now

        # flow residuals
        residuals['q'] = solver.getResidual(ap)


    def solve_nonlinear(self, inputs, outputs):

        solver = self.options['solver']
        ap = self.options['ap']

        if self._do_solve:

            # Set the warped mesh
            #solver.mesh.setSolverGrid(inputs['x_g'])
            # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now

            self._set_ap(inputs)
            ap.solveFailed = False # might need to clear this out?
            ap.fatalFail = False

            solver(ap)

            #if ap.fatalFail:
            #    if self.comm.rank == 0:
            #        print('###############################################################')
            #        print('#Solve Fatal Fail. Analysis Error')
            #        print('###############################################################')

            #    raise AnalysisError('ADFLOW Solver Fatal Fail')


            #if ap.solveFailed: # the mesh was fine, but it didn't converge
            #    if self.comm.rank == 0:
            #        print('###############################################################')
            #        print('#Solve Failed, attempting a clean restart!')
            #        print('###############################################################')

            #    ap.solveFailed = False
            #    ap.fatalFail = False
            #    solver.resetFlow(ap)
            #    solver(ap)

            #    if ap.solveFailed or ap.fatalFail: # we tried, but there was no saving it
            #        print('###############################################################')
            #        print('#Clean Restart failed. There is no saving this one!')
            #        print('###############################################################')

            #        raise AnalysisError('ADFLOW Solver Fatal Fail')

        outputs['q'] = solver.getStates()


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
                    wDeriv=True, xVDeriv=True, xDvDeriv=True)

                if 'q' in d_outputs:
                    d_outputs['q'] += wBar

                if 'x_g' in d_inputs:
                    d_inputs['x_g'] += xVBar

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

        self.distributed = True

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

        # Set the warped mesh
        #solver.mesh.setSolverGrid(inputs['x_g'])
        # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now
        self._set_states(inputs)

        outputs['f_a'] = solver.getForces().flatten(order='C')

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        solver = self.options['solver']
        ap = self.options['ap']

        if mode == 'fwd':
            if 'f_a' in d_outputs:
                if 'q' in d_inputs:
                    wDot = d_inputs['q']
                else:
                    wDot = None
                if 'x_g' in d_inputs:
                    xVDot = d_inputs['x_g']
                else:
                    xVDot = None
                if not(xVDot is None and wDot is None):
                    dfdot = solver.computeJacobianVectorProductFwd(xVDot=xVDot,
                                                                   wDot=wDot,
                                                                   fDeriv=True)
                    d_outputs['f_a'] += dfdot.flatten()

        elif mode == 'rev':
            if 'f_a' in d_outputs:
                fBar = d_outputs['f_a']
                #print ('fBar',fBar)

                wBar, xVBar, xDVBar = solver.computeJacobianVectorProductBwd(
                    fBar=fBar,
                    wDeriv=True, xVDeriv=True, xDvDeriv=True)

                if 'x_g' in d_inputs:
                    d_inputs['x_g'] += xVBar
                    #print ('xVBor',xVBar)
                if 'q' in d_inputs:
                    d_inputs['q'] += wBar
                    #print ('wBor',wBar)

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

class AdflowFunctions(ExplicitComponent):

    def initialize(self):
        self.options.declare('ap', types=AeroProblem,)
        self.options.declare('solver')

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True


    def setup(self):
        solver = self.options['solver']
        ap = self.options['ap']

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            value = 1.
            if 'value' in kwargs:
                value = kwargs['value']

            self.add_input(name, shape=size, units=kwargs['units'])

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

        for f_name in self.options['ap'].evalFuncs:
        #for f_name, f_meta in solver.adflowCostFunctions.items():
        #    f_type = f_meta[1]
        #    units = None
        #    if f_type in FUNCS_UNITS:
        #        units = FUNCS_UNITS[f_type]

            if self.comm.rank == 0:
                print("adding adflow func as output: {}".format(f_name))
            self.add_output(f_name, shape=1)
            #self.add_output(f_name, shape=1, units=units)

            #self.declare_partials(of=f_name, wrt='*')

    def _set_ap(self, inputs):
        tmp = {}
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            tmp[name] = inputs[name][0]

        self.options['ap'].setDesignVars(tmp)
        #self.options['solver'].setAeroProblem(self.options['ap'])

    def _set_states(self, inputs):
        self.options['solver'].setStates(inputs['q'])

    def _get_func_name(self, name):
        return '%s_%s' % (self.options['ap'].name, name.lower())

    def compute(self, inputs, outputs):
        solver = self.options['solver']
        ap = self.options['ap']
        #print('funcs compute')
        #actually setting things here triggers some kind of reset, so we only do it if you're actually solving
        if self._do_solve:
            self._set_ap(inputs)
            # Set the warped mesh
            #solver.mesh.setSolverGrid(inputs['x_g'])
            # ^ This call does not exist. Assume the mesh hasn't changed since the last call to the warping comp for now
            self._set_states(inputs)

        funcs = {}

        eval_funcs = [f_name for f_name in self.options['ap'].evalFuncs]
        solver.evalFunctions(ap, funcs, eval_funcs)
        #solver.evalFunctions(ap, funcs)

        #for name in ap.evalFuncs:
        for name in self.options['ap'].evalFuncs:
            f_name = self._get_func_name(name)
            if f_name in funcs:
                outputs[name.lower()] = funcs[f_name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        solver = self.options['solver']
        ap = self.options['ap']


        #self.options['solver'].setAeroProblem(ap)
        #print('func matvec')

        #if self._do_solve:
        #    self._set_ap(inputs)
        #    self._set_geo(inputs)
        #    self._set_states(inputs)

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

            for name  in self.options['ap'].evalFuncs:
                func_name = name.lower()

                # we have to check for 0 here, so we don't include any unnecessary variables in funcsBar
                # becasue it causes ADflow to do extra work internally even if you give it extra variables, even if the seed is 0
                if func_name in d_outputs and d_outputs[func_name] != 0.:
                    funcsBar[func_name] = d_outputs[func_name][0] / self.comm.size

            # because of the 0 checking, the funcsBar is now only correct on the root proc,
            # so we need to broadcast it to everyone. its not actually important that the seeds are the same everywhere,
            # but the keys in the dictionary need to be the same.
            funcsBar = self.comm.bcast(funcsBar, root=0)


            #print(funcsBar, flush=True)

            d_input_vars = list(d_inputs.keys())
            n_input_vars = len(d_input_vars)

            wBar = None
            xVBar = None
            xDVBar = None

            wBar, xVBar, xDVBar = solver.computeJacobianVectorProductBwd(
                funcsBar=funcsBar,
                wDeriv=True, xVDeriv=True, xDvDeriv=True)
            if 'q' in d_inputs:
                d_inputs['q'] += wBar
            if 'x_g' in d_inputs:
                d_inputs['x_g'] += xVBar

            for dv_name, dv_bar in xDVBar.items():
                if dv_name in d_inputs:
                    d_inputs[dv_name] += dv_bar.flatten()
