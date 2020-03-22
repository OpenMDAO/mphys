
import numpy as np
import pprint

from baseclasses import AeroProblem

from adflow import ADFLOW
from idwarp import USMesh

from openmdao.api import Group, ImplicitComponent, ExplicitComponent

from adflow.python.om_utils import get_dvs_and_cons

class AdflowMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates

    """
    def initialize(self):
        self.options.declare('aero_solver')
        self.options['distributed'] = True

    def setup(self):

        self.aero_solver = self.options['aero_solver']

        self.x_a0 = self.aero_solver.mesh.getSurfaceCoordinates().flatten(order='C')

        coord_size = self.x_a0.size

        self.add_output('x_a0_mesh', shape=coord_size, desc='initial aerodynamic surface node coordinates')

    def compute(self,inputs,outputs):
        outputs['x_a0_mesh'] = self.x_a0

class Geo_Disp(ExplicitComponent):
    """
    This component adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options['distributed'] = True
        self.options.declare('nnodes')

    def setup(self):
        aero_nnodes = self.options['nnodes']
        local_size = aero_nnodes * 3
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

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

class AdflowWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.

    """

    def initialize(self):
        self.options.declare('aero_solver')
        #self.options.declare('use_OM_KSP', default=False, types=bool,
        #    desc="uses OpenMDAO's PestcKSP linear solver with ADflow's preconditioner to solve the adjoint.")

        self.options['distributed'] = True

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options['aero_solver']
        # self.add_output('foo', val=1.0)
        solver = self.solver

        # self.ap_vars,_ = get_dvs_and_cons(ap=ap)

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
        self.options.declare('aero_solver')
        #self.options.declare('use_OM_KSP', default=False, types=bool,
        #    desc="uses OpenMDAO's PestcKSP linear solver with ADflow's preconditioner to solve the adjoint.")

        self.options['distributed'] = True

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options['aero_solver']
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

        #self.declare_partials(of='q', wrt='*')

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
                    wDeriv=True, xVDeriv=True, xDvDeriv=True)

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

class AdflowForces(ExplicitComponent):
    """
    OpenMDAO component that wraps force integration

    """

    def initialize(self):
        self.options.declare('aero_solver')

        self.options['distributed'] = True

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options['aero_solver']
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
                    wDeriv=True, xVDeriv=True, xDvDeriv=True)

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

class AdflowFunctions(ExplicitComponent):

    def initialize(self):
        self.options.declare('aero_solver')

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True


    def setup(self):

        self.solver = self.options['aero_solver']
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

class ADflow_group(Group):

    def initialize(self):
        self.options.declare('solver')
        self.options.declare('as_coupling')

    def setup(self):

        self.aero_solver = self.options['solver']
        self.as_coupling = self.options['as_coupling']

        if self.as_coupling:
            self.add_subsystem('geo_disp', Geo_Disp(
                nnodes=int(self.aero_solver.getSurfaceCoordinates().size /3)),
                promotes_inputs=['u_a', 'x_a0']
            )
        # if we dont have geo_disp, we also need to promote the x_a as x_a0 from the deformer component
        self.add_subsystem('deformer', AdflowWarper(
            aero_solver=self.aero_solver
        ))
        self.add_subsystem('solver', AdflowSolver(
            aero_solver=self.aero_solver
        ))
        if self.as_coupling:
            self.add_subsystem('force', AdflowForces(
                aero_solver=self.aero_solver),
                promotes_outputs=['f_a']
            )
        self.add_subsystem('funcs', AdflowFunctions(
            aero_solver=self.aero_solver
        ))

    def configure(self):

        self.connect('deformer.x_g', ['solver.x_g', 'funcs.x_g'])
        self.connect('solver.q', 'funcs.q')

        if self.as_coupling:
            self.connect('geo_disp.x_a', 'deformer.x_a')
            self.connect('deformer.x_g', 'force.x_g')
            self.connect('solver.q', 'force.q')
        else:
            self.promotes('deformer', inputs=[('x_a', 'x_a0')])


    def mphy_set_ap(self, ap):
        # set the ap, add inputs and outputs, promote?
        self.solver.set_ap(ap)
        self.funcs.set_ap(ap)
        if self.as_coupling:
            self.force.set_ap(ap)

        # promote the DVs for this ap
        ap_vars,_ = get_dvs_and_cons(ap=ap)

        for (args, kwargs) in ap_vars:
            name = args[0]
            size = args[1]
            self.promotes('solver', inputs=[name])
            self.promotes('funcs', inputs=[name])
            if self.as_coupling:
                self.promotes('force', inputs=[name])

class ADflow_builder(object):

    def __init__(self, options):
        self.options = options

    # api level method for all builders
    def init_solver(self, comm):
        self.solver = ADFLOW(options=self.options, comm=comm)
        mesh = USMesh(options=self.options)
        self.solver.setMesh(mesh)

    # api level method for all builders
    def get_solver(self):
        return self.solver

    # api level method for all builders
    def get_element(self, **kwargs):
        return ADflow_group(solver=self.solver, **kwargs)

    def get_mesh_element(self):
        return AdflowMesh(aero_solver=self.solver)

    def get_nnodes(self):
        return int(self.solver.getSurfaceCoordinates().size /3)