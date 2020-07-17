
import numpy as np
from pprint import pprint as pp
from mpi4py import MPI

from baseclasses import AeroProblem

from adflow import ADFLOW
from idwarp import USMesh

from openmdao.api import Group, ImplicitComponent, ExplicitComponent, AnalysisError

from adflow.om_utils import get_dvs_and_cons

class ADflowMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates

    """
    def initialize(self):
        self.options.declare('aero_solver', recordable=False)
        self.options['distributed'] = True

    def setup(self):

        self.aero_solver = self.options['aero_solver']

        self.x_a0 = self.aero_solver.getSurfaceCoordinates(includeZipper=False).flatten(order='C')
        # self.x_a0 = self.aero_solver.mesh.getSurfaceCoordinates().flatten(order='C')

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

    def mphys_get_triangulated_surface(self, groupName=None):
        # this is a list of lists of 3 points
        # p0, v1, v2

        return self._getTriangulatedMeshSurface()

    def _getTriangulatedMeshSurface(self, groupName=None, **kwargs):
        """
        This function returns a trianguled verision of the surface
        mesh on all processors. The intent is to use this for doing
        constraints in DVConstraints.

        Returns
        -------
        surf : list
           List of points and vectors describing the surface. This may
           be passed directly to DVConstraint setSurface() function.
        """

        if groupName is None:
            groupName = self.aero_solver.allWallsGroup

        # Obtain the points and connectivity for the specified
        # groupName
        pts = self.aero_solver.comm.allgather(self.aero_solver.getSurfaceCoordinates(groupName, **kwargs))
        conn, faceSizes = self.aero_solver.getSurfaceConnectivity(groupName)
        conn = np.array(conn).flatten()
        conn = self.aero_solver.comm.allgather(conn)
        faceSizes = self.aero_solver.comm.allgather(faceSizes)

        # Triangle info...point and two vectors
        p0 = []
        v1 = []
        v2 = []

        # loop over the faces
        for iProc in range(len(faceSizes)):

            connCounter=0
            for iFace in range(len(faceSizes[iProc])):
                # Get the number of nodes on this face
                faceSize = faceSizes[iProc][iFace]
                faceNodes = conn[iProc][connCounter:connCounter+faceSize]

                # Start by getting the centerpoint
                ptSum= [0, 0, 0]
                for i in range(faceSize):
                    #idx = ptCounter+i
                    idx = faceNodes[i]
                    ptSum+=pts[iProc][idx]

                avgPt = ptSum/faceSize

                # Now go around the face and add a triangle for each adjacent pair
                # of points. This assumes an ordered connectivity from the
                # meshwarping
                for i in range(faceSize):
                    idx = faceNodes[i]
                    p0.append(avgPt)
                    v1.append(pts[iProc][idx]-avgPt)
                    if(i<(faceSize-1)):
                        idxp1 = faceNodes[i+1]
                        v2.append(pts[iProc][idxp1]-avgPt)
                    else:
                        # wrap back to the first point for the last element
                        idx0 = faceNodes[0]
                        v2.append(pts[iProc][idx0]-avgPt)

                # Now increment the connectivity
                connCounter+=faceSize

        return [p0, v1, v2]

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

class GeoDisp(ExplicitComponent):
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

class ADflowWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.

    """

    def initialize(self):
        self.options.declare('aero_solver', recordable=False)
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
        local_coord_size = solver.getSurfaceCoordinates(includeZipper=False).size
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
                    dxS = self.solver.mapVector(dxS, self.solver.meshFamilyGroup,
                                                self.solver.designFamilyGroup, includeZipper=False)
                    d_inputs['x_a'] += dxS.flatten()

class ADflowSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the ADflow flow solver

    """

    def initialize(self):
        self.options.declare('aero_solver', recordable=False)
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

        # this is the solution counter for failed solution outputs.
        # the converged solutions are written by the adflow functionals group
        self.solution_counter = 0

        # flag to keep track if the current solution started from a clean restart,
        # or it was restarted from the previous converged state.
        self.cleanRestart = True

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

        # enable if you want to print all aero dv inputs
        # if self.comm.rank == 0:
        #     print('aero dv inputs:')
        #     pp(tmp)

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
                print('%s (%s)'%(name, kwargs['units']))

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

            # do not write solution files inside the solver loop
            solver(ap, writeSolution=False)

            if ap.fatalFail:
                if self.comm.rank == 0:
                    print('###############################################################')
                    print('# Solve Fatal Fail. Analysis Error')
                    print('###############################################################')

                raise AnalysisError('ADFLOW Solver Fatal Fail')


            if ap.solveFailed: # the mesh was fine, but it didn't converge
                # if the previous iteration was already a clean restart, dont try again
                if self.cleanRestart:
                    if self.comm.rank == 0:
                        print('###############################################################')
                        print('# This was a clean restart. Will not try another one.')
                        print('###############################################################')

                    # write the solution so that we can diagnose
                    solver.writeSolution(baseName='analysis_fail' ,number=self.solution_counter)
                    self.solution_counter += 1

                    solver.resetFlow(ap)
                    self.cleanRestart = True
                    raise AnalysisError('ADFLOW Solver Fatal Fail')

                # the previous iteration restarted from another solution, so we can try again
                # with a re-set flowfield for the initial guess.
                else:
                    if self.comm.rank == 0:
                        print('###############################################################')
                        print('# Solve Failed, attempting a clean restart!')
                        print('###############################################################')

                    # write the solution so that we can diagnose
                    solver.writeSolution(baseName='analysis_fail' ,number=self.solution_counter)
                    self.solution_counter += 1

                    ap.solveFailed = False
                    ap.fatalFail = False
                    solver.resetFlow(ap)
                    solver(ap, writeSolution=False)

                    if ap.solveFailed or ap.fatalFail: # we tried, but there was no saving it
                        if self.comm.rank == 0:
                            print('###############################################################')
                            print('# Clean Restart failed. There is no saving this one!')
                            print('###############################################################')

                        # write the solution so that we can diagnose
                        solver.writeSolution(baseName='analysis_fail' ,number=self.solution_counter)
                        self.solution_counter += 1

                        # re-set the flow for the next iteration:
                        solver.resetFlow(ap)
                        # set the reset flow flag
                        self.cleanRestart = True
                        raise AnalysisError('ADFLOW Solver Fatal Fail')

                    # see comment for the same flag below
                    else:
                        self.cleanRestart = False

            # solve did not fail, therefore we will re-use this converged flowfield for the next iteration.
            # change the flag so that if the next iteration fails with current initial guess, it can retry
            # with a clean restart
            else:
                self.cleanRestart = False

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

class ADflowForces(ExplicitComponent):
    """
    OpenMDAO component that wraps force integration

    """

    def initialize(self):
        self.options.declare('aero_solver', recordable=False)

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
        # if self.comm.rank == 0:
        #     print('adding ap var inputs:')
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            self.add_input(name, shape=size, units=kwargs['units'])
            # if self.comm.rank == 0:
            #     print('%s (%s)'%(name, kwargs['units']))

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
    'aavgptot': 'Pa',
    'aavgps': 'Pa',
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

class ADflowFunctions(ExplicitComponent):

    def initialize(self):
        self.options.declare('aero_solver', recordable=False)
        # flag to automatically add the AP functions as output
        self.options.declare('ap_funcs', default=True)
        self.options.declare('write_solution', default=True)

        # testing flag used for unit-testing to prevent the call to actually solve
        # NOT INTENDED FOR USERS!!! FOR TESTING ONLY
        self._do_solve = True

        self.prop_funcs = None

    def setup(self):

        self.solver = self.options['aero_solver']
        self.ap_funcs = self.options['ap_funcs']
        self.write_solution = self.options['write_solution']
        solver = self.solver
        #self.set_check_partial_options(wrt='*',directional=True)
        self.solution_counter = 0

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

    def mphys_set_ap(self, ap):
        # this is the external function to set the ap to this component
        self.ap = ap

        self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # parameter inputs
        # if self.comm.rank == 0:
            # print('adding ap var inputs:')
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]
            self.add_input(name, shape=size, units=kwargs['units'])
            # if self.comm.rank == 0:
                # print('%s with units %s'%(name, kwargs['units']))

        if self.ap_funcs:
            if self.comm.rank == 0:
                print("adding adflow funcs as output:")

            for f_name in sorted(list(self.ap.evalFuncs)):
                # get the function type. this is the first word before the first underscore
                f_type = f_name.split('_')[0]

                # check if we have a unit defined for this
                if f_type in FUNCS_UNITS:
                    units = FUNCS_UNITS[f_type]
                else:
                    units = None

                # print the function name and units
                if self.comm.rank == 0:
                    print("%s (%s)"%(f_name, units))

                self.add_output(f_name, shape=1, units=units)

    def mphys_add_prop_funcs(self, prop_funcs):
        # save this list
        self.prop_funcs = prop_funcs

        # loop over the functions here and create the output

        if self.comm.rank == 0:
            print("adding adflow funcs as propulsion output:")

        for f_name in prop_funcs:
            # get the function type. this is the first word before the first underscore
            f_type = f_name.split('_')[0]

            # check if we have a unit defined for this
            if f_type in FUNCS_UNITS:
                units = FUNCS_UNITS[f_type]
            else:
                units = None

            # print the function name and units
            if self.comm.rank == 0:
                print("%s (%s)"%(f_name, units))

            self.add_output(f_name, shape=1, units=units)

    def _set_states(self, inputs):
        self.solver.setStates(inputs['q'])

    def _get_func_name(self, name):
        return '%s_%s' % (self.ap.name, name.lower())

    def nom_write_solution(self):
        # this writes the solution files and is callable from outside openmdao call routines
        solver = self.solver
        ap = self.ap

        # re-set the AP so that we are sure state is updated
        solver.setAeroProblem(ap)

        # write the solution files. Internally, this checks the
        # types of solution files specified in the options and
        # only outsputs these
        solver.writeSolution(number=self.solution_counter)
        self.solution_counter += 1

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

        if self.write_solution:
            # write the solution files. Internally, this checks the
            # types of solution files specified in the options and
            # only outsputs these
            solver.writeSolution(number=self.solution_counter)
            self.solution_counter += 1

        funcs = {}

        if self.ap_funcs:
            # without the sorted, each proc might get a different order...
            eval_funcs = sorted(list(self.ap.evalFuncs))
            solver.evalFunctions(ap, funcs, evalFuncs=eval_funcs)

            for name in self.ap.evalFuncs:
                f_name = self._get_func_name(name)
                if f_name in funcs:
                    outputs[name.lower()] = funcs[f_name]

        if self.prop_funcs is not None:
            # also do the prop
            solver.evalFunctions(ap, funcs, evalFuncs=self.prop_funcs)
            for name in self.prop_funcs:
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

            if self.ap_funcs:
                for name  in self.ap.evalFuncs:
                    func_name = name.lower()

                    # we have to check for 0 here, so we don't include any unnecessary variables in funcsBar
                    # becasue it causes ADflow to do extra work internally even if you give it extra variables, even if the seed is 0
                    if func_name in d_outputs and d_outputs[func_name] != 0.:
                        funcsBar[func_name] = d_outputs[func_name][0]
                        # this stuff is fixed now. no need to divide
                        # funcsBar[func_name] = d_outputs[func_name][0] / self.comm.size
                        # print(self.comm.rank, func_name, funcsBar[func_name])

            # also do the same for prop functions
            if self.prop_funcs is not None:
                for name  in self.prop_funcs:
                    func_name = name.lower()
                    if func_name in d_outputs and d_outputs[func_name] != 0.:
                        funcsBar[func_name] = d_outputs[func_name][0]

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

class ADflowGroup(Group):

    def initialize(self):
        self.options.declare('solver')
        self.options.declare('as_coupling')
        # TODO remove the default
        self.options.declare('prop_coupling', default=False)
        self.options.declare('use_warper', default=True)
        self.options.declare('balance_group', default=None)

    def setup(self):

        self.aero_solver = self.options['solver']
        self.as_coupling = self.options['as_coupling']
        self.prop_coupling = self.options['prop_coupling']

        self.use_warper = self.options['use_warper']

        balance_group = self.options['balance_group']

        if self.as_coupling:
            self.add_subsystem('geo_disp', GeoDisp(
                nnodes=int(self.aero_solver.getSurfaceCoordinates().size /3)),
                promotes_inputs=['u_a', 'x_a0']
            )
        if self.use_warper:
            # if we dont have geo_disp, we also need to promote the x_a as x_a0 from the deformer component
            self.add_subsystem('deformer',
                ADflowWarper(
                    aero_solver=self.aero_solver,
                ),
                promotes_outputs=['x_g'],
            )

        self.add_subsystem('solver',
            ADflowSolver(
                aero_solver=self.aero_solver,
            ),
            promotes_inputs=['x_g'],
        )

        if self.as_coupling:
            self.add_subsystem('force', ADflowForces(
                aero_solver=self.aero_solver),
                promotes_inputs=['x_g'],
                promotes_outputs=['f_a'],
            )
        if self.prop_coupling:
            self.add_subsystem('prop',
                ADflowFunctions(
                    aero_solver=self.aero_solver,
                    ap_funcs=False,
                    write_solution=False,
                ),
                promotes_inputs=['x_g'],
            )

        if balance_group is not None:
            self.add_subsystem('balance', balance_group)

    def configure(self):

        if self.as_coupling:
            self.connect('geo_disp.x_a', 'deformer.x_a')
        else:
            if self.use_warper:
                self.promotes('deformer', inputs=[('x_a', 'x_a0')])

        if self.as_coupling:
            self.connect('solver.q', 'force.q')

        if self.prop_coupling:
            self.connect('solver.q', 'prop.q')

        # TODO if we have a balance, automatically figure out the broyden stuff here

    def mphys_set_ap(self, ap):
        # set the ap, add inputs and outputs, promote?
        self.solver.set_ap(ap)
        # self.funcs.set_ap(ap)
        if self.as_coupling:
            self.force.set_ap(ap)
        if self.prop_coupling:
            self.prop.mphys_set_ap(ap)

        # promote the DVs for this ap
        ap_vars,_ = get_dvs_and_cons(ap=ap)

        for (args, kwargs) in ap_vars:
            name = args[0]
            size = args[1]
            self.promotes('solver', inputs=[name])
            # self.promotes('funcs', inputs=[name])
            if self.as_coupling:
                self.promotes('force', inputs=[name])
            if self.prop_coupling:
                self.promotes('prop', inputs=[name])

    def mphys_add_prop_funcs(self, prop_funcs):
        # this is the main routine to enable outputs from the propulsion element

        # call the method of the prop element
        self.prop.mphys_add_prop_funcs(prop_funcs)

        # promote these variables to the aero group level
        self.promotes('prop', outputs=prop_funcs)

class ADflowMeshGroup(Group):

    def initialize(self):
        self.options.declare('aero_solver')

    def setup(self):
        aero_solver = self.options['aero_solver']

        self.add_subsystem('surface_mesh',  ADflowMesh(aero_solver=aero_solver), promotes=['*'])
        self.add_subsystem('volume_mesh',
            ADflowWarper(
                aero_solver=aero_solver
            ),
            promotes_inputs=[('x_a', 'x_a0')],
            promotes_outputs=['x_g'],
        )

    def mphys_add_coordinate_input(self):
        # just pass through the call
        return self.surface_mesh.mphys_add_coordinate_input()

    def mphys_get_triangulated_surface(self):
        # just pass through the call
        return self.surface_mesh.mphys_get_triangulated_surface()

class ADflowBuilder(object):

    def __init__(self, options, warp_in_solver=True, balance_group=None, prop_coupling=False):
        self.options = options
        self.warp_in_solver = warp_in_solver
        self.prop_coupling = prop_coupling

        self.balance_group = balance_group

    # api level method for all builders
    def init_solver(self, comm):
        self.solver = ADFLOW(options=self.options, comm=comm)
        mesh = USMesh(options=self.options, comm=comm)
        self.solver.setMesh(mesh)

    # api level method for all builders
    def get_solver(self):
        return self.solver

    # api level method for all builders
    def get_element(self, **kwargs):
        use_warper = self.warp_in_solver
        return ADflowGroup(solver=self.solver, use_warper=use_warper, balance_group=self.balance_group, prop_coupling=self.prop_coupling, **kwargs)

    def get_mesh_element(self):
        use_warper = not self.warp_in_solver
        # if we do warper in the mesh element, we will do a group thing
        if use_warper:
            return ADflowMeshGroup(aero_solver=self.solver)
        else:
            return ADflowMesh(aero_solver=self.solver)

    def get_scenario_element(self):
        return ADflowFunctions(aero_solver=self.solver)

    def get_scenario_connections(self):
        # this is the stuff we want to be connected
        # between the solver and the functionals.
        # these variables FROM the solver are connected
        # TO the funcs element. So the solver has output
        # and funcs has input. key is the output,
        # variable is the input in the returned dict.
        if self.warp_in_solver:
            mydict = {
                'x_g': 'x_g',
                'solver.q': 'q',
            }
        else:
            mydict = {
                'solver.q': 'q',
            }

        return mydict

    def get_mesh_connections(self):
        if self.warp_in_solver:
            mydict = {
                'solver':{
                    'x_a0'  : 'x_a0',
                },
                'funcs':{},
            }
        else:
            mydict = {
                'solver':{
                    'x_g'  : 'x_g',
                },
                'funcs':{
                    'x_g'  : 'x_g',
                },
            }

        return mydict

    def get_nnodes(self):
        return int(self.solver.getSurfaceCoordinates().size /3)
