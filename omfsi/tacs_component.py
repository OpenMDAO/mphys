from __future__ import division, print_function
import numpy as np
from mpi4py import MPI

from openmdao.api import ImplicitComponent, ExplicitComponent, IndepVarComp, ParallelGroup
from tacs import TACS,functions


class StructuralGroup(ParallelGroup):
    """
    The structural parallel group wraps the TACS components to allow them to
    operate on a subset of the available procs if desired
    """
    def initialize(self):
        self.options.declare('struct_comp', desc='Structural solver component')
        self.options.declare('nprocs',default=1,desc='number of structural processors')

    def setup(self):
        struct_comp = self.options['struct_comp']
        nprocs      = self.options['nprocs']

        self.add_subsystem('struct_comp', struct_comp, promotes =['*']
                                                     , max_procs=nprocs)
class TacsMesh(ExplicitComponent):
    """
    Component to read the initial mesh coordinates with TACS

    Assumptions:
        - User will provide a tacs_setup function that assigns tacs elements to the mesh
          => tacs = tacs_mesh_setup(comm)

    """
    def initialize(self):
        self.options.declare('tacs_mesh_setup', default = None, desc='Function to setup tacs')
        self.options['distributed'] = True

    def setup(self):

        # TACS assembler setup
        tacs_setup = self.options['tacs_mesh_setup']
        tacs = tacs_setup(self.comm)

        self.tacs = tacs

        # create some TACS bvecs that will be needed later
        self.xpts  = tacs.createNodeVec()
        self.tacs.getNodes(self.xpts)

        # OpenMDAO setup
        node_size  =     self.xpts.getArray().size
        self.add_output('x_s', shape=node_size, desc='structural node coordinates')

    def compute(self,inputs,outputs):
        outputs['x_s'] = self.xpts.getArray()

class TacsSolver(ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - User will provide a tacs_solver_setup function that gives some pieces
          required for the tacs solver
          => tacs, mat, pc, gmres, struct_ndv = tacs_solver_setup(comm)
        - The TACS steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('tacs_solver_setup', default = None, desc='Function to setup tacs')
        self.options['distributed'] = True

        self.tacs = None
        self.pc = None


        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.psi_s = None
        self.x_save = None

        self.transposed = False

    def setup(self):

        # TACS assembler setup
        tacs_solver_setup = self.options['tacs_solver_setup']
        tacs, mat, pc, gmres, ndv = tacs_solver_setup(self.comm)

        self.tacs = tacs
        self.pc = pc
        self.gmres = gmres
        self.mat = mat

        # create some TACS bvecs that will be needed later
        self.res        = tacs.createVec()
        self.force      = tacs.createVec()
        self.ans        = tacs.createVec()
        self.struct_rhs = tacs.createVec()
        self.psi_s      = tacs.createVec()
        self.xpt_sens   = tacs.createNodeVec()

        # OpenMDAO setup
        xpts = tacs.createNodeVec()

        state_size = self.ans.getArray().size
        node_size  =     xpts.getArray().size

        # inputs
        self.add_input('dv_struct', shape=ndv       , desc='tacs design variables')
        self.add_input('x_s',       shape=node_size , desc='structural node coordinates')
        self.add_input('f_s',       shape=state_size, desc='structural load vector')

        # outputs
        self.add_output('u_s',      shape=state_size, desc='structural state vector')

        # partials
        self.declare_partials('u_s',['dv_struct','x_s','f_s'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        tacs = self.tacs
        res  = self.res
        ans  = self.ans

        if self._design_vector_changed(inputs['dv_struct']) or self.transposed:
            pc     = self.pc
            tacs.setDesignVars(inputs['dv_struct'])
            alpha = 1.0
            beta  = 0.0
            gamma = 0.0

            res = self.res
            res_array = res.getArray()
            res_array[:] = 0.0

            tacs.assembleJacobian(alpha,beta,gamma,res,self.mat)
            pc.factor()
            self.transposed = False

        ans_array = ans.getArray()
        ans_array[:] = outputs['u_s']
        tacs.setVariables(ans)

        res_array = res.getArray()
        res_array[:] = 0.0

        # K * u
        tacs.assembleRes(res)

        # Add the external loads
        res_array[:] -= inputs['f_s']

        # Apply BCs to the residual
        tacs.applyBCs(res)

        residuals['u_s'][:] = res_array[:]

    def solve_nonlinear(self, inputs, outputs):
        tacs   = self.tacs
        force  = self.force
        ans    = self.ans
        pc     = self.pc
        gmres  = self.gmres

        # if the design variables changed or we're coming in with a transposed
        # matrix, update the stiffness matrix
        if self._design_vector_changed(inputs['dv_struct']) or self.transposed:
            tacs.setDesignVars(inputs['dv_struct'])
            alpha = 1.0
            beta  = 0.0
            gamma = 0.0

            res = self.res
            res_array = res.getArray()
            res_array[:] = 0.0

            tacs.assembleJacobian(alpha,beta,gamma,res,self.mat)
            pc.factor()
            self.transposed = False

        # solve the linear system
        force_array = force.getArray()
        force_array[:] = inputs['f_s']
        tacs.applyBCs(force)

        gmres.solve(force, ans)
        ans_array = ans.getArray()
        outputs['u_s'] = ans_array[:]

    def solve_linear(self,d_outputs,d_residuals,mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
                tacs = self.tacs
                gmres = self.gmres

                # if nonsymmetric, we need to form the transpose Jacobian
                #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
                #    alpha = 1.0
                #    beta  = 0.0
                #    gamma = 0.0

                #    res = self.res
                #    res_array = res.getArray()
                #    res_array[:] = 0.0
                #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
                #    pc.factor()
                #    self.transposed=True

                res = self.res
                res_array = res.getArray()
                res_array[:] = d_outputs['u_s']
                psi_s = self.psi_s
                gmres.solve(res,psi_s)
                psi_s_array = psi_s.getArray()
                d_residuals['u_s'] = psi_s_array

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_s' in d_residuals:
                tacs = self.tacs

                res  = self.res
                res_array = res.getArray()

                ans  = self.ans
                ans_array = ans.getArray()

                if 'u_s' in d_outputs:

                    ans_array[:] = outputs['u_s']
                    tacs.setVariables(ans)

                    # if nonsymmetric, we need to form the transpose Jacobian
                    #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
                    #    alpha = 1.0
                    #    beta  = 0.0
                    #    gamma = 0.0
                    #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
                    #    self.transposed=True

                    res_array[:] = 0.0

                    self.mat.mult(ans,res)

                    d_outputs['u_s'] += res_array[:]

                if 'f_s' in d_inputs:
                    # dR/df_s^T = -I
                    d_inputs['f_s'] -= d_residuals['u_s']

                if 'x_s' in d_inputs:
                    ans_array[:] = d_residuals['u_s']
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()

                    tacs.evalAdjointResXptSensProduct(ans, xpt_sens)

                    d_inputs['x_s'] += xpt_sens_array[:]

                if 'dv_struct' in d_inputs:
                    adjResProduct  = np.zeros(d_inputs['dv_struct'].size)
                    psi_s_array    = self.psi_s.getArray()
                    psi_s_array[:] = d_residuals['u_s']
                    self.tacs.evalAdjointResProduct(self.psi_s, adjResProduct)
                    d_inputs['dv_struct'] +=  adjResProduct

    def _design_vector_changed(self,x):
        if self.x_save is None:
            self.x_save = x.copy()
            return True
        elif not np.allclose(x,self.x_save,rtol=1e-10,atol=1e-10):
            self.x_save = x.copy()
            return True
        else:
            return False


class TacsFunctions(ExplicitComponent):
    """
    Component to compute TACS functions

    Assumptions:
        - User will provide a tacs_func_setup function that will set up a list of functions
          => func_list, tacs, struct_ndv = tacs_func_setup(comm)
    """
    def initialize(self):
        self.options.declare('tacs_func_setup', desc = 'function to feed tacs function-evaluation information')

        self.ans = None
        self.tacs = None

        self.mass = False

    def setup(self):

        # TACS part of setup
        tacs_func_setup = self.options['tacs_func_setup']
        func_list, tacs, ndv = tacs_func_setup(self.comm)

        self.tacs = tacs

        self.ans = tacs.createVec()
        state_shape = self.ans.getArray().size

        self.xpt_sens = tacs.createNodeVec()
        xpts_shape = self.xpt_sens.getArray().size

        # OpenMDAO part of setup
        self.add_input('dv_struct', shape=ndv,            desc='tacs design variables')
        self.add_input('x_s',       shape=xpts_shape,     desc='structural node coordinates')
        self.add_input('u_s',       shape=state_shape,    desc='structural state vector')

        # Remove the mass function from the func list if it is there
        # since it is not dependent on the structural state
        func_no_mass = []
        for i,func in enumerate(func_list):
            if isinstance(func,functions.StructuralMass):
                if not self.mass:
                    self.add_output('mass', 0.0, desc = 'structural mass')
                    self.mass = True
            else:
                func_no_mass.append(func)

        self.func_list = func_no_mass
        self.add_output('f_struct', shape=len(self.func_list), desc='structural function values')

        # declare the partials
        self.declare_partials('f_struct',['dv_struct','x_s','u_s'])
        if self.mass:
            self.declare_partials('mass',['dv_struct','x_s'])

    def compute(self,inputs,outputs):

        ans = self.ans
        ans_array = ans.getArray()
        ans_array[:] = inputs['u_s']

        self.tacs.setVariables(ans)

        outputs['f_struct'] = self.tacs.evalFunctions(self.func_list)

        if 'mass' in outputs:
            func = functions.StructuralMass(self.tacs)
            outputs['mass'] = self.tacs.evalFunctions([func])

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if 'mass' in d_outputs:
                func = functions.StructuralMass(self.tacs)
                if 'dv_struct' in d_inputs:
                    # get df/dx if the function is a structural function
                    size = d_inputs['dv_struct'].size
                    dvsens = np.zeros(d_inputs['dv_struct'].size)
                    self.tacs.evalDVSens(func, dvsens)

                    d_inputs['dv_struct'] += dvsens

                if 'x_s' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()
                    self.tacs.evalXptSens(func, xpt_sens)

                    d_inputs['x_s'] += xpt_sens_array

            if 'f_struct' in d_outputs:
                ans = self.ans
                for ifunc, func in enumerate(self.func_list):
                    if 'dv_struct' in d_inputs:
                        dvsens = np.zeros(d_inputs['dv_struct'].size)
                        self.tacs.evalDVSens(func, dvsens)

                        d_inputs['dv_struct'][:] += dvsens

                    if 'x_s' in d_inputs:
                        xpt_sens = self.xpt_sens
                        xpt_sens_array = xpt_sens.getArray()
                        self.tacs.evalXptSens(func, xpt_sens)

                        d_inputs['x_s'][:] += xpt_sens_array

                    if 'u_s' in d_inputs:
                        self.tacs.evalSVSens(func,ans)
                        ans_array = ans.getArray()

                        d_inputs['u_s'][:] += ans_array

class PrescribedLoad(ExplicitComponent):
    """
    Prescribe a load to tacs

    Assumptions:
        - User will provide a load_function prescribes the loads
          => load = load_function(x_s,ndof)
        - User will provide a get_tacs function
          => tacs = get_tacs()
    """
    def initialize(self):
        self.options.declare('load_function', default = None, desc='function that prescribes the loads')
        self.options.declare('get_tacs', default = None, desc='function that gets tacs')

        self.options['distributed'] = True

        self.ndof = 0

    def setup(self):

        # TACS assembler setup
        tacs = self.options['get_tacs']()

        # create some TACS vectors so we can see what size they are
        xpts  = tacs.createNodeVec()
        node_size = xpts.getArray().size

        tmp   = tacs.createVec()
        state_size = tmp.getArray().size
        self.ndof = int(state_size / ( node_size / 3 ))

        # OpenMDAO setup
        self.add_input('x_s', shape=node_size, desc='structural load')
        self.add_output('f_s', shape=state_size, desc='structural load')

        self.declare_partials('f_s','x_s')

    def compute(self,inputs,outputs):
        load_function = self.options['load_function']
        outputs['f_s'] = load_function(inputs['x_s'],self.ndof)

