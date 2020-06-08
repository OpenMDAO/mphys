from __future__ import division, print_function
from builder_class import Builder
import openmdao.api as om
from tacs import TACS,functions
import numpy as np

class TacsMesh(om.ExplicitComponent):
    """
    Component to read the initial mesh coordinates with TACS

    """
    def initialize(self):
        # self.options.declare('get_tacs', default = None, desc='function to get tacs')
        self.options.declare('struct_solver', default = None, desc='the tacs object itself')
        self.options.declare('surface_nodes', default = None, desc='surface nodes')
        self.options['distributed'] = True

    def setup(self):

        tacs = self.options['struct_solver']
                # create some TACS bvecs that will be needed later
        self.xpts  = tacs.createNodeVec()
        tacs.getNodes(self.xpts)

        # OpenMDAO setup
        node_size  =     self.xpts.getArray().size
        print('mesher x_s0', node_size)

        self.surface_nodes = self.options['surface_nodes']
        self.add_output('x_s0', shape=node_size, desc='structural node coordinates')

        self.add_output('x_s0_surface', shape= self.surface_nodes.size, desc='structural node coordinates')

    def mphys_add_coordinate_input(self):
        local_size  = self.xpts.getArray().size
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_s0_points', shape=node_size, src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')

        # return the promoted name and coordinates
        return 'x_s0_points', self.xpts.getArray()

    def compute(self,inputs,outputs):
        if 'x_s0_points' in inputs:
            outputs['x_s0'] = inputs['x_s0_points']
        else:

            outputs['x_s0'] = self.xpts.getArray()
            outputs['x_s0_surface'] = self.surface_nodes

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'x_s0_points' in d_inputs:
                d_outputs['x_s0'] += d_inputs['x_s0_points']
        elif mode == 'rev':
            if 'x_s0_points' in d_inputs:
                d_inputs['x_s0_points'] += d_outputs['x_s0']

class TacsSolver(om.ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - User will provide a tacs_solver_setup function that gives some pieces
          required for the tacs solver
          => tacs, mat, pc, gmres, struct_ndv = tacs_solver_setup(comm)
        - The TACS steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('struct_solver')
        self.options.declare('struct_objects')
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.tacs = None
        self.pc = None

        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.psi_s = None
        self.x_save = None

        self.transposed = False
        self.check_partials = False

    def setup(self):
        self.check_partials = self.options['check_partials']

        tacs = self.options['struct_solver']
        struct_objects = self.options['struct_objects']
        # these objects come from self.struct_objects but ideally, they should be attributes of the struct solver object
        mat = struct_objects[0]
        pc = struct_objects[1]
        gmres = struct_objects[2]
        ndv = struct_objects[3]['ndv']
        self.solver_dict = struct_objects[3]

        # TACS assembler setup
        self.tacs      = tacs
        self.mat       = mat
        self.pc        = pc
        self.gmres     = gmres
        self.ndv       = ndv

        # create some TACS bvecs that will be needed later
        self.res        = tacs.createVec()
        self.force      = tacs.createVec()
        self.ans        = tacs.createVec()
        self.struct_rhs = tacs.createVec()
        self.psi_s      = tacs.createVec()
        self.xpt_sens   = tacs.createNodeVec()

        # OpenMDAO setup

        state_size = self.ans.getArray().size
        node_size  = self.xpt_sens.getArray().size
        self.ndof = int(state_size/(node_size/3))

        s_list = self.comm.allgather(state_size)
        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])


        # inputs
        self.add_input('dv_struct', shape=ndv                                                 , desc='tacs design variables')
        self.add_input('x_s0',      shape=node_size , src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')
        self.add_input('f_s',       shape=state_size, src_indices=np.arange(s1, s2, dtype=int), desc='structural load vector')

        # outputs
        self.add_output('u_s',      shape=state_size, val = np.zeros(state_size),desc='structural state vector')

        # partials
        #self.declare_partials('u_s',['dv_struct','x_s0','f_s'])

    def get_ndof(self):
        return self.solver_dict['ndof']

    def get_nnodes(self):
        return self.solver_dict['nnodes']

    def get_ndv(self):
        return self.solver_dict['ndv']

    def get_funcs(self):
        return self.solver_dict['get_funcs']

    def _update_internal(self,inputs,outputs=None):
        self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        pc     = self.pc
        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        res = self.tacs.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self.tacs.assembleJacobian(alpha,beta,gamma,res,self.mat)
        pc.factor()

        if outputs is not None:
            ans = self.ans
            ans_array = ans.getArray()
            ans_array[:] = outputs['u_s']
            self.tacs.applyBCs(ans)

            self.tacs.setVariables(ans)

    def apply_nonlinear(self, inputs, outputs, residuals):
        tacs = self.tacs
        res  = self.res
        ans  = self.ans

        self._update_internal(inputs,outputs)

        res_array = res.getArray()
        res_array[:] = 0.0

        # K * u
        tacs.assembleRes(res)

        # Add the external loads
        res_array[:] -= inputs['f_s']

        # Apply BCs to the residual (forces)
        tacs.applyBCs(res)

        residuals['u_s'][:] = res_array[:]

        



    def solve_nonlinear(self, inputs, outputs):
        tacs   = self.tacs
        force  = self.force
        ans    = self.ans
        pc     = self.pc
        gmres  = self.gmres

        self._update_internal(inputs)
        # solve the linear system
        force_array = force.getArray()
        force_array[:] = inputs['f_s']
        tacs.applyBCs(force)

        gmres.solve(force, ans)
        ans_array = ans.getArray()
        outputs['u_s'] = ans_array[:]
        tacs.setVariables(ans)

    def solve_linear(self,d_outputs,d_residuals,mode):
        if mode == 'fwd':
            if self.check_partials:
                print ('solver fwd')
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            tacs = self.tacs
            gmres = self.gmres

            # if nonsymmetric, we need to form the transpose Jacobian
            #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
            #    alpha = 1.0
            #    beta  = 0.0
            #    gamma = 0.0

            #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
            #    pc.factor()
            #    self.transposed=True

            res = self.res
            res_array = res.getArray()
            res_array[:] = d_outputs['u_s']
            before = res_array.copy()
            tacs.applyBCs(res)
            after = res_array.copy()
            psi_s = self.psi_s
            gmres.solve(res,psi_s)
            psi_s_array = psi_s.getArray()
            tacs.applyBCs(psi_s)
            d_residuals['u_s'] = psi_s_array.copy()
            d_residuals['u_s'] -= np.array(after - before,dtype=np.float64)

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        self._update_internal(inputs,outputs)
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_s' in d_residuals:
                tacs = self.tacs

                res  = self.res
                res_array = res.getArray()

                ans  = self.ans
                ans_array = ans.getArray()

                psi = tacs.createVec()
                psi_array = psi.getArray()
                psi_array[:] = d_residuals['u_s'][:]

                before = psi_array.copy()
                tacs.applyBCs(psi)
                after = psi_array.copy()

                if 'u_s' in d_outputs:

                    ans_array[:] = outputs['u_s']
                    tacs.applyBCs(ans)
                    tacs.setVariables(ans)

                    # if nonsymmetric, we need to form the transpose Jacobian
                    #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
                    #    alpha = 1.0
                    #    beta  = 0.0
                    #    gamma = 0.0
                    #    tacs.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
                    #    pc.factor()
                    #    self.transposed=True

                    res_array[:] = 0.0

                    self.mat.mult(psi,res)
                    # tacs.applyBCs(res)

                    d_outputs['u_s'] += np.array(res_array[:],dtype=float)
                    d_outputs['u_s'] -= np.array(after - before,dtype=np.float64)

                if 'f_s' in d_inputs:
                    d_inputs['f_s'] -= np.array(psi_array[:],dtype=float)

                if 'x_s0' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()

                    tacs.evalAdjointResXptSensProduct(psi, xpt_sens)

                    d_inputs['x_s0'] += np.array(xpt_sens_array[:],dtype=float)

                if 'dv_struct' in d_inputs:
                    adj_res_product  = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                    self.tacs.evalAdjointResProduct(psi, adj_res_product)

                    # TACS has already done a parallel sum (mpi allreduce) so
                    # only add the product on one rank
                    if self.comm.rank == 0:
                        d_inputs['dv_struct'] +=  np.array(adj_res_product,dtype=float)

    def _design_vector_changed(self,x):
        if self.x_save is None:
            self.x_save = x.copy()
            return True
        elif not np.allclose(x,self.x_save,rtol=1e-10,atol=1e-10):
            self.x_save = x.copy()
            return True
        else:
            return False


class TacsSolver_Conduction(om.ImplicitComponent):
    """
    Component to perform TACS steady conduction analysis

    Assumptions:
        - User will provide a tacs_solver_setup function that gives some pieces
          required for the tacs solver
          => tacs, mat, pc, gmres, struct_ndv = tacs_solver_setup(comm)
        - The TACS steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('struct_solver')
        self.options.declare('struct_objects')
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.tacs = None
        self.pc = None

        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.psi_s = None
        self.x_save = None

        self.transposed = False
        self.check_partials = False

    def setup(self):
        self.check_partials = self.options['check_partials']

        tacs = self.options['struct_solver']
        struct_objects = self.options['struct_objects']
        # these objects come from self.struct_objects but ideally, they should be attributes of the struct solver object
        mat = struct_objects[0]
        pc = struct_objects[1]
        gmres = struct_objects[2]
        ndv = struct_objects[3]['ndv']
        self.solver_dict = struct_objects[3]

        # TACS assembler setup
        self.tacs      = tacs
        self.mat       = mat
        self.pc        = pc
        self.gmres     = gmres
        self.ndv       = ndv

        # create some TACS bvecs that will be needed later
        self.res        = tacs.createVec()
        self.force      = tacs.createVec()
        self.ans        = tacs.createVec()
        self.heat       = tacs.createVec()
        self.struct_rhs = tacs.createVec()
        self.psi_s      = tacs.createVec()
        self.xpt_sens   = tacs.createNodeVec()

        # OpenMDAO setup
        surface_nodes = self.solver_dict['surface_nodes']

        self.mapping = self.solver_dict['mapping']

        node_size  = self.xpt_sens.getArray().size
        # self.ndof = int(state_size/(node_size/3))

        s_list = self.comm.allgather(surface_nodes.size//3)
        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])
        


        # inputs
        # self.add_input('dv_struct', shape=ndv                                                 , desc='tacs design variables')
        print('conduction x_s0', node_size)
        self.add_input('x_s0',      shape=node_size , src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')

        print('conduction heat_xfer', surface_nodes.size/3)
        self.add_input('heat_xfer',       shape=surface_nodes.size//3, src_indices=np.arange(s1, s2, dtype=int), desc='structural load vector')

        # outputs
        print('conduction temp_cond', surface_nodes.size/3)
        self.add_output('temp_cond',      shape=surface_nodes.size//3, val = np.ones(surface_nodes.size//3)*300,desc='temperature vector')

        # partials
        #self.declare_partials('u_s',['dv_struct','x_s0','f_s'])

    def get_ndof(self):
        return self.solver_dict['ndof']

    def get_nnodes(self):
        return self.solver_dict['nnodes']

    def get_ndv(self):
        return self.solver_dict['ndv']

    def get_funcs(self):
        return self.solver_dict['get_funcs']

    def _update_internal(self,inputs,outputs=None):
        # self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        pc     = self.pc
        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        res = self.tacs.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self.tacs.assembleJacobian(alpha,beta,gamma,res,self.mat)
        pc.factor()
        

    def solve_nonlinear(self, inputs, outputs):
        tacs   = self.tacs
        force  = self.force
        ans    = self.ans
        pc     = self.pc
        gmres  = self.gmres

        self._update_internal(inputs)
        heat = self.heat
        heat_array = heat.getArray()
        
        # may need to do mapping here
        for i in range(len(self.mapping)):
            heat_array[self.mapping[i]] = inputs['heat_xfer'][i]


        self.tacs.setBCs(heat)


        gmres.solve(heat, ans)
        ans_array = ans.getArray()
        tacs.setVariables(ans)


        ans_array = ans.getArray()
    
        # get specifically the temps from the nodes in the mapping
        # i.e. the surface nodes of the structure
        for i in range(len(self.mapping)):
            outputs['temp_cond'][i] = ans_array[self.mapping[i]]


    


class TacsFunctions(om.ExplicitComponent):
    """
    Component to compute TACS functions

    Assumptions:
        - User will provide a tacs_func_setup function that will set up a list of functions
          => func_list, tacs, struct_ndv = tacs_func_setup(comm)
    """
    def initialize(self):
        self.options.declare('struct_solver')
        self.options.declare('struct_objects')
        self.options.declare('check_partials')

        self.ans = None
        self.tacs = None

        self.check_partials = False

    def setup(self):

        self.tacs = self.options['struct_solver']
        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        ndv = self.struct_objects[3]['ndv']
        get_funcs = self.struct_objects[3]['get_funcs']

        tacs = self.tacs

        func_list = get_funcs(tacs)

        # TACS part of setup
        self.ndv       = ndv
        self.func_list = func_list

        self.ans = tacs.createVec()
        state_size = self.ans.getArray().size

        self.xpt_sens = tacs.createNodeVec()
        node_size = self.xpt_sens.getArray().size

        s_list = self.comm.allgather(state_size)
        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        s1 = np.sum(s_list[:irank])
        s2 = np.sum(s_list[:irank+1])
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        # OpenMDAO part of setup
        # TODO move the dv_struct to an external call where we add the DVs
        self.add_input('dv_struct', shape=ndv,                                                    desc='tacs design variables')
        self.add_input('x_s0',      shape=node_size,  src_indices=np.arange(n1, n2, dtype=int),   desc='structural node coordinates')
        self.add_input('u_s',       shape=state_size, src_indices=np.arange(s1, s2, dtype=int),   desc='structural state vector')

        # Remove the mass function from the func list if it is there
        # since it is not dependent on the structural state
        func_no_mass = []
        for i,func in enumerate(func_list):
            if not isinstance(func,functions.StructuralMass):
                func_no_mass.append(func)

        self.func_list = func_no_mass
        if len(self.func_list) > 0:
            self.add_output('f_struct', shape=len(self.func_list), desc='structural function values')

            # declare the partials
            #self.declare_partials('f_struct',['dv_struct','x_s0','u_s'])

    def _update_internal(self,inputs):
        self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        mat    = self.tacs.createFEMat()
        pc     = TACS.Pc(mat)
        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

        res = self.tacs.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self.tacs.assembleJacobian(alpha,beta,gamma,res,mat)
        pc.factor()

        ans = self.ans
        ans_array = ans.getArray()
        ans_array[:] = inputs['u_s']
        self.tacs.applyBCs(ans)

        self.tacs.setVariables(ans)

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'f_struct' in outputs:
            outputs['f_struct'] = self.tacs.evalFunctions(self.func_list)

        # TODO fix this with the configure based approach
        # if self.options['f5_writer'] is not None:
        #     self.options['f5_writer'](self.tacs)

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if self.check_partials:
                self._update_internal(inputs)

            if 'f_struct' in d_outputs:
                for ifunc, func in enumerate(self.func_list):
                    self.tacs.evalFunctions([func])
                    if 'dv_struct' in d_inputs:
                        dvsens = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                        self.tacs.evalDVSens(func, dvsens)

                        d_inputs['dv_struct'][:] += np.array(dvsens,dtype=float) * d_outputs['f_struct'][ifunc]

                    if 'x_s0' in d_inputs:
                        xpt_sens = self.xpt_sens
                        xpt_sens_array = xpt_sens.getArray()
                        self.tacs.evalXptSens(func, xpt_sens)

                        d_inputs['x_s0'][:] += np.array(xpt_sens_array,dtype=float) * d_outputs['f_struct'][ifunc]

                    if 'u_s' in d_inputs:
                        prod = self.tacs.createVec()
                        self.tacs.evalSVSens(func,prod)
                        prod_array = prod.getArray()

                        d_inputs['u_s'][:] += np.array(prod_array,dtype=float) * d_outputs['f_struct'][ifunc]
class TacsMass(om.ExplicitComponent):
    """
    Component to compute TACS mass

    Assumptions:
        - User will provide a tacs_func_setup function that will set up a list of functions
          => func_list, tacs, struct_ndv = tacs_func_setup(comm)
    """
    def initialize(self):
        self.options.declare('struct_solver')
        self.options.declare('struct_objects')
        self.options.declare('check_partials')

        self.ans = None
        self.tacs = None

        self.mass = False

        self.check_partials = False

    def setup(self):

        self.tacs = self.options['struct_solver']
        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        # self.set_check_partial_options(wrt='*',directional=True)

        tacs = self.tacs

        # TACS part of setup
        self.tacs = tacs
        ndv  = self.struct_objects[3]['ndv']

        self.xpt_sens = tacs.createNodeVec()
        node_size = self.xpt_sens.getArray().size

        n_list = self.comm.allgather(node_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        # OpenMDAO part of setup
        self.add_input('dv_struct', shape=ndv,                                                    desc='tacs design variables')
        self.add_input('x_s0',      shape=node_size,  src_indices=np.arange(n1, n2, dtype=int),   desc='structural node coordinates')

        self.add_output('mass', 0.0, desc = 'structural mass')
        #self.declare_partials('mass',['dv_struct','x_s0'])

    def _update_internal(self,inputs):
        self.tacs.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs.createNodeVec()
        self.tacs.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_s0']
        self.tacs.setNodes(xpts)

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'mass' in outputs:
            func = functions.StructuralMass(self.tacs)
            outputs['mass'] = self.tacs.evalFunctions([func])

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('forward mode requested but not implemented')
        if mode == 'rev':
            if self.check_partials:
                self._update_internal(inputs)
            if 'mass' in d_outputs:
                func = functions.StructuralMass(self.tacs)
                if 'dv_struct' in d_inputs:
                    size = d_inputs['dv_struct'].size
                    dvsens = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                    self.tacs.evalDVSens(func, dvsens)

                    d_inputs['dv_struct'] += np.array(dvsens,dtype=float) * d_outputs['mass']

                if 'x_s0' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()
                    self.tacs.evalXptSens(func, xpt_sens)

                    d_inputs['x_s0'] += np.array(xpt_sens_array,dtype=float) * d_outputs['mass']


class PrescribedLoad(om.ExplicitComponent):
    """
    Prescribe a load to tacs

    Assumptions:
        - User will provide a load_function prescribes the loads
          => load = load_function(x_s0,ndof)
        - User will provide a get_tacs function
          => tacs = get_tacs()
    """
    def initialize(self):
        self.options.declare('load_function', default = None, desc='function that prescribes the loads')
        self.options.declare('tacs')

        self.options['distributed'] = True

        self.ndof = 0

    def setup(self):
#        self.set_check_partial_options(wrt='*',directional=True)

        # TACS assembler setup
        tacs = self.options['tacs']

        # create some TACS vectors so we can see what size they are
        # TODO getting the node sizes should be easier than this...
        xpts  = tacs.createNodeVec()
        node_size = xpts.getArray().size

        tmp   = tacs.createVec()
        state_size = tmp.getArray().size
        self.ndof = int(state_size / ( node_size / 3 ))

        irank = self.comm.rank

        n_list = self.comm.allgather(node_size)
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        # OpenMDAO setup
        self.add_input('x_s0', shape=node_size, src_indices=np.arange(n1, n2, dtype=int), desc='structural node coordinates')
        self.add_output('f_s', shape=state_size, desc='structural load')

        #self.declare_partials('f_s','x_s0')

    def compute(self,inputs,outputs):
        load_function = self.options['load_function']
        outputs['f_s'] = load_function(inputs['x_s0'],self.ndof)

class TACS_group(om.Group):
    def initialize(self):
        self.options.declare('solver')
        self.options.declare('solver_objects')
        self.options.declare('check_partials')
        self.options.declare('conduction')
        

    def setup(self):
        self.struct_solver = self.options['solver']
        self.struct_objects = self.options['solver_objects']
        self.check_partials = self.options['check_partials']

        # check if we have a loading function
        solver_dict = self.struct_objects[3]

        if 'load_function' in solver_dict:
            self.prescribed_load = True
            self.add_subsystem('loads', PrescribedLoad(
                load_function=solver_dict['load_function'],
                tacs=self.struct_solver
            ), promotes_inputs=['x_s0'], promotes_outputs=['f_s'])

        if self.options['conduction']:           
            self.add_subsystem('solver', TacsSolver_Conduction(
                struct_solver=self.struct_solver,
                struct_objects=self.struct_objects,
                check_partials=self.check_partials),
                promotes_inputs=['heat_xfer', 'x_s0'],
                promotes_outputs=['temp_cond']
            )
        else:            
            self.add_subsystem('solver', TacsSolver(
                struct_solver=self.struct_solver,
                struct_objects=self.struct_objects,
                check_partials=self.check_partials),
                promotes_inputs=['f_s', 'x_s0', 'dv_struct'],
                promotes_outputs=['u_s']
            )

            self.add_subsystem('funcs', TacsFunctions(
                struct_solver=self.struct_solver,
                struct_objects=self.struct_objects,
                check_partials=self.check_partials),
                promotes_inputs=['x_s0', 'dv_struct']
            )

            self.add_subsystem('mass', TacsMass(
                struct_solver=self.struct_solver,
                struct_objects=self.struct_objects,
                check_partials=self.check_partials),
                promotes_inputs=['x_s0', 'dv_struct']
            )

    def configure(self):
        if not self.options['conduction']:           
            self.connect('u_s', 'funcs.u_s')

class TACS_builder(Builder):

    def __init__(self, options,check_partials=False, conduction=False):
        super(TACS_builder, self).__init__(options)
        self.check_partials = check_partials
        self.conduction = conduction


    # api level method for all builders
    def init_solver(self, comm):

        solver_dict={}

        mesh = TACS.MeshLoader(comm)
        mesh.scanBDFFile(self.options['mesh_file'])

        ndof, ndv = self.options['add_elements'](mesh)
        self.n_dv_struct = ndv

        tacs = mesh.createTACS(ndof)

        nnodes = int(tacs.createNodeVec().getArray().size / 3)

        if self.conduction:
            mat = tacs.createSchurMat()
        else:
            mat = tacs.createFEMat()

        pc = TACS.Pc(mat)

        subspace = 100
        restarts = 2
        gmres = TACS.KSM(mat, pc, subspace, restarts)

        solver_dict['ndv']    = ndv
        solver_dict['ndof']   = ndof
        solver_dict['nnodes'] = nnodes
        solver_dict['get_funcs'] = self.options['get_funcs']
        
        #use the supplied function to get the surface points and mapping
        if self.conduction:
            solver_dict['surface_nodes'], solver_dict['mapping'] = self.options['get_surface'](tacs)
            
        # check if the user provided a load function
        if 'load_function' in self.options:
            solver_dict['load_function'] = self.options['load_function']

        self.solver_dict=solver_dict

        # put the rest of the stuff in a tuple
        solver_objects = [mat, pc, gmres, solver_dict]

        self.solver = tacs
        self.solver_objects = solver_objects

    # api level method for all builders
    def get_solver(self):
        return self.solver

    # api level method for all builders
    def get_element(self, **kwargs):
        return TACS_group(solver=self.solver, solver_objects=self.solver_objects,check_partials=self.check_partials, **kwargs)

    def get_mesh_element(self):
        return TacsMesh(struct_solver=self.solver)

    def get_ndof(self):
        return self.solver_dict['ndof']

    def get_nnodes(self):
        return self.solver_dict['nnodes']

    def get_ndv(self):
        return self.solver_dict['ndv']


    def get_object(self):
        return self.solver
    
    def build_object(self, comm):
        self.init_solver(comm)

    def get_component(self, **kwargs):
        yield '_mesh', TacsMesh(struct_solver=self.solver, surface_nodes=self.solver_dict['surface_nodes'])
        yield '', TACS_group(solver=self.solver, solver_objects=self.solver_objects, check_partials=self.check_partials, conduction=self.conduction, **kwargs)
        # yield 'hi', 0

