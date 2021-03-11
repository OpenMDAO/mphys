import numpy as np
from tacs import TACS,functions

import openmdao.api as om
from mphys.builder import Builder

class TacsMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates with TACS
    """
    def initialize(self):
        self.options.declare('tacs_assembler', default = None, desc='the tacs object itself', recordable=False)
        self.options['distributed'] = True

    def setup(self):
        tacs_assembler = self.options['tacs_assembler']
        xpts = tacs_assembler.createNodeVec()
        x = xpts.getArray()
        tacs_assembler.getNodes(xpts)
        self.add_output('x_struct0', val=x, shape=x.size, desc='structural node coordinates', tags=['mphys_coordinates'])

class TacsSolver(om.ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - The TACS steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('tacs_assembler', recordable=False)
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.tacs_assembler = None
        self.pc = None

        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.psi_s = None
        self.x_save = None

        self.transposed = False
        self.check_partials = False

        self.old_dvs = None
        self.old_xs = None

    def setup(self):
        self.check_partials = self.options['check_partials']
        #self.set_check_partial_options(wrt='*',method='cs',directional=True)

        tacs_assembler = self.options['tacs_assembler']
        struct_objects = self.options['struct_objects']
        # these objects come from self.struct_objects but ideally, they should be attributes of the struct solver object
        mat = struct_objects[0]
        pc = struct_objects[1]
        gmres = struct_objects[2]
        ndv = struct_objects[3]['ndv']
        self.solver_dict = struct_objects[3]

        # TACS assembler setup
        self.tacs_assembler      = tacs_assembler
        self.mat       = mat
        self.pc        = pc
        self.gmres     = gmres
        self.ndv       = ndv

        # create some TACS bvecs that will be needed later
        self.res        = tacs_assembler.createVec()
        self.force      = tacs_assembler.createVec()
        self.ans        = tacs_assembler.createVec()
        self.struct_rhs = tacs_assembler.createVec()
        self.psi_s      = tacs_assembler.createVec()
        self.xpt_sens   = tacs_assembler.createNodeVec()

        # OpenMDAO setup
        state_size = self.ans.getArray().size
        node_size  = self.xpt_sens.getArray().size
        self.ndof = int(state_size/(node_size/3))

        # inputs
        self.add_input('dv_struct', shape=ndv, desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', shape_by_conn=True, desc='structural node coordinates',tags=['mphys_coordinates'])
        self.add_input('f_struct',  shape_by_conn=True, desc='structural load vector', tags=['mphys_coupling'])

        # outputs
        # its important that we set this to zero since this displacement value is used for the first iteration of the aero
        self.add_output('u_struct', shape=state_size, val = np.zeros(state_size),desc='structural state vector', tags=['mphys_coupling'])

        # partials
        #self.declare_partials('u_struct',['dv_struct','x_struct0','f_struct'])

    def _need_update(self,inputs):

        update = False

        if self.old_dvs is None:
            self.old_dvs = inputs['dv_struct'].copy()
            update =  True

        for dv, dv_old in zip(inputs['dv_struct'],self.old_dvs):
            if np.abs(dv - dv_old) > 0.:#1e-7:
                self.old_dvs = inputs['dv_struct'].copy()
                update =  True

        if self.old_xs is None:
            self.old_xs = inputs['x_struct0'].copy()
            update =  True

        for xs, xs_old in zip(inputs['x_struct0'],self.old_xs):
            if np.abs(xs - xs_old) > 0.:#1e-7:
                self.old_xs = inputs['x_struct0'].copy()
                update =  True

        return update

    def _update_internal(self,inputs,outputs=None):
        if self._need_update(inputs):
            self.tacs_assembler.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

            xpts = self.tacs_assembler.createNodeVec()
            self.tacs_assembler.getNodes(xpts)
            xpts_array = xpts.getArray()
            xpts_array[:] = inputs['x_struct0']
            self.tacs_assembler.setNodes(xpts)

            pc     = self.pc
            alpha = 1.0
            beta  = 0.0
            gamma = 0.0

            res = self.tacs_assembler.createVec()
            res_array = res.getArray()
            res_array[:] = 0.0

            self.tacs_assembler.assembleJacobian(alpha,beta,gamma,res,self.mat)
            pc.factor()

        if outputs is not None:
            ans = self.ans
            ans_array = ans.getArray()
            ans_array[:] = outputs['u_struct']
            self.tacs_assembler.applyBCs(ans)

            self.tacs_assembler.setVariables(ans)

    def apply_nonlinear(self, inputs, outputs, residuals):
        tacs_assembler = self.tacs_assembler
        res  = self.res
        ans  = self.ans

        self._update_internal(inputs,outputs)

        res_array = res.getArray()
        res_array[:] = 0.0

        # K * u
        tacs_assembler.assembleRes(res)

        # Add the external loads
        res_array[:] -= inputs['f_struct']

        # Apply BCs to the residual (forces)
        tacs_assembler.applyBCs(res)

        residuals['u_struct'][:] = res_array[:]

    def solve_nonlinear(self, inputs, outputs):

        tacs_assembler   = self.tacs_assembler
        force  = self.force
        ans    = self.ans
        pc     = self.pc
        gmres  = self.gmres

        self._update_internal(inputs)
        # solve the linear system
        force_array = force.getArray()
        force_array[:] = inputs['f_struct']
        tacs_assembler.applyBCs(force)

        gmres.solve(force, ans)
        ans_array = ans.getArray()
        outputs['u_struct'] = ans_array[:]
        tacs_assembler.setVariables(ans)

    def solve_linear(self,d_outputs,d_residuals,mode):

        if mode == 'fwd':
            if self.check_partials:
                print ('solver fwd')
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            tacs_assembler = self.tacs_assembler
            gmres = self.gmres

            # if nonsymmetric, we need to form the transpose Jacobian
            #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
            #    alpha = 1.0
            #    beta  = 0.0
            #    gamma = 0.0

            #    tacs_assembler.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
            #    pc.factor()
            #    self.transposed=True

            res = self.res
            res_array = res.getArray()
            res_array[:] = d_outputs['u_struct']

            # Tacs doesn't actually transpose the matrix here so keep track of
            # RHS entries that TACS zeros out for BCs that openmdao is not
            # aware of.
            before = res_array.copy()
            tacs_assembler.applyBCs(res)
            after = res_array.copy()

            gmres.solve(res,self.psi_s)
            psi_s_array = self.psi_s.getArray()
            tacs_assembler.applyBCs(self.psi_s)
            d_residuals['u_struct'] = psi_s_array.copy()
            d_residuals['u_struct'] -= np.array(after - before,dtype=np.float64)

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        self._update_internal(inputs,outputs)
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('TACS forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_struct' in d_residuals:
                tacs_assembler = self.tacs_assembler

                res  = self.res
                res_array = res.getArray()

                ans  = self.ans
                ans_array = ans.getArray()

                psi = tacs_assembler.createVec()
                psi_array = psi.getArray()
                psi_array[:] = d_residuals['u_struct'][:]

                before = psi_array.copy()
                tacs_assembler.applyBCs(psi)
                after = psi_array.copy()

                if 'u_struct' in d_outputs:

                    ans_array[:] = outputs['u_struct']
                    tacs_assembler.applyBCs(ans)
                    tacs_assembler.setVariables(ans)

                    # if nonsymmetric, we need to form the transpose Jacobian
                    #if self._design_vector_changed(inputs['dv_struct']) or not self.transposed:
                    #    alpha = 1.0
                    #    beta  = 0.0
                    #    gamma = 0.0
                    #    tacs_assembler.assembleJacobian(alpha,beta,gamma,res,self.mat,matOr=TACS.PY_TRANSPOSE)
                    #    pc.factor()
                    #    self.transposed=True

                    res_array[:] = 0.0

                    self.mat.mult(psi,res)
                    # tacs_assembler.applyBCs(res)

                    d_outputs['u_struct'] += np.array(res_array[:],dtype=float)
                    d_outputs['u_struct'] -= np.array(after - before,dtype=np.float64)

                if 'f_struct' in d_inputs:
                    d_inputs['f_struct'] -= np.array(psi_array[:],dtype=float)

                if 'x_struct0' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()

                    tacs_assembler.evalAdjointResXptSensProduct(psi, xpt_sens)

                    d_inputs['x_struct0'] += np.array(xpt_sens_array[:],dtype=float)

                if 'dv_struct' in d_inputs:
                    adj_res_product  = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                    self.tacs_assembler.evalAdjointResProduct(psi, adj_res_product)

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


class TacsSolverConduction(om.ImplicitComponent):
    """
    Component to perform TACS steady conduction analysis

    Assumptions:
        - User will provide a tacs_solver_setup function that gives some pieces
          required for the tacs solver
          => tacs, mat, pc, gmres, struct_ndv = tacs_solver_setup(comm)
        - The TACS steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('tacs_assembler')
        self.options.declare('struct_objects')
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.tacs_assembler = None
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

        self.tacs_assembler = self.options['tacs_assembler']
        struct_objects = self.options['struct_objects']
        # these objects come from self.struct_objects but ideally, they should be attributes of the struct solver object
        self.mat = struct_objects[0]
        self.pc = struct_objects[1]
        self.gmres = struct_objects[2]
        self.ndv = struct_objects[3]['ndv']
        self.solver_dict = struct_objects[3]

        # create some TACS bvecs that will be needed later
        self.res        = self.tacs_assembler.createVec()
        self.force      = self.tacs_assembler.createVec()
        self.ans        = self.tacs_assembler.createVec()
        self.heat       = self.tacs_assembler.createVec()
        self.struct_rhs = self.tacs_assembler.createVec()
        self.psi_s      = self.tacs_assembler.createVec()
        self.xpt_sens   = self.tacs_assembler.createNodeVec()

        # OpenMDAO setup
        surface_nodes = self.solver_dict['surface_nodes']

        self.mapping = self.solver_dict['mapping']

        # self.ndof = int(state_size/(node_size/3))


        # inputs
        # self.add_input('dv_struct', shape=ndv                                                 , desc='tacs design variables')
        self.add_input('x_struct0', shape_by_conn=True, desc='structural node coordinates', tags=['mphys_coordinates'])
        self.add_input('q_conduct',  shape_by_conn=True, desc='structural load vector', tags=['mphys_coupling'])

        # outputs
        self.add_output('T_conduct',      shape=surface_nodes.size//3, val = np.ones(surface_nodes.size//3)*300,desc='temperature vector', tags=['mphys_coupling'])

        # partials
        #self.declare_partials('u_struct',['dv_struct','x_struct0','f_struct'])

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

        xpts = self.tacs_assembler.createNodeVec()
        self.tacs_assembler.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_struct0']
        self.tacs_assembler.setNodes(xpts)

        res = self.tacs_assembler.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self.tacs_assembler.assembleJacobian(alpha,beta,gamma,res,self.mat)
        pc.factor()

    def solve_nonlinear(self, inputs, outputs):
        ans    = self.ans
        gmres  = self.gmres

        self._update_internal(inputs)
        heat = self.heat
        heat_array = heat.getArray()

        # may need to do mapping here
        for i in range(len(self.mapping)):
            heat_array[self.mapping[i]] = inputs['q_conduct'][i]

        self.tacs_assembler.setBCs(heat)

        gmres.solve(heat, ans)
        ans_array = ans.getArray()
        self.tacs_assembler.setVariables(ans)

        ans_array = ans.getArray()

        # get specifically the temps from the nodes in the mapping
        # i.e. the surface nodes of the structure
        for i in range(len(self.mapping)):
            outputs['T_conduct'][i] = ans_array[self.mapping[i]]


class TacsFunctions(om.ExplicitComponent):
    """
    Component to compute TACS functions
    """
    def initialize(self):
        self.options.declare('tacs_assembler', recordable=False)
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.ans = None
        self.tacs_assembler = None

        self.check_partials = False

    def setup(self):

        self.tacs_assembler = self.options['tacs_assembler']
        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        ndv = self.struct_objects[3]['ndv']
        get_funcs = self.struct_objects[3]['get_funcs']

        if 'f5_writer' in self.struct_objects[3].keys():
            self.f5_writer = self.struct_objects[3]['f5_writer']
        else:
            self.f5_writer = None

        tacs_assembler = self.tacs_assembler

        func_list = get_funcs(tacs_assembler)

        # TACS part of setup
        self.ndv       = ndv
        self.func_list = func_list

        self.ans = tacs_assembler.createVec()

        self.xpt_sens = tacs_assembler.createNodeVec()

        # OpenMDAO part of setup
        # TODO move the dv_struct to an external call where we add the DVs
        self.add_input('dv_struct', shape_by_conn=True, desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', shape_by_conn=True, desc='structural node coordinates',tags=['mphys_coordinates'])
        self.add_input('u_struct',  shape_by_conn=True, desc='structural state vector', tags=['mphys_coupling'])

        # Remove the mass function from the func list if it is there
        # since it is not dependent on the structural state
        func_no_mass = []
        for i,func in enumerate(func_list):
            if not isinstance(func,functions.StructuralMass):
                func_no_mass.append(func)

        self.func_list = func_no_mass
        if len(self.func_list) > 0:
            self.add_output('func_struct', shape=len(self.func_list), desc='structural function values', tags=['mphys_result'])

            # declare the partials
            #self.declare_partials('f_struct',['dv_struct','x_struct0','u_struct'])

    def _update_internal(self,inputs):
        self.tacs_assembler.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs_assembler.createNodeVec()
        self.tacs_assembler.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_struct0']
        self.tacs_assembler.setNodes(xpts)

        mat    = self.tacs_assembler.createFEMat()
        pc     = TACS.Pc(mat)
        alpha = 1.0
        beta  = 0.0
        gamma = 0.0

        res = self.tacs_assembler.createVec()
        res_array = res.getArray()
        res_array[:] = 0.0

        self.tacs_assembler.assembleJacobian(alpha,beta,gamma,res,mat)
        pc.factor()

        ans = self.ans
        ans_array = ans.getArray()
        ans_array[:] = inputs['u_struct']
        self.tacs_assembler.applyBCs(ans)

        self.tacs_assembler.setVariables(ans)

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'func_struct' in outputs:
            outputs['func_struct'] = self.tacs_assembler.evalFunctions(self.func_list)

        if self.f5_writer is not None:
            self.f5_writer(self.tacs_assembler)

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('TACS forward mode requested but not implemented')
        if mode == 'rev':
            if self.check_partials:
                self._update_internal(inputs)

            if 'func_struct' in d_outputs:
                for ifunc, func in enumerate(self.func_list):
                    self.tacs_assembler.evalFunctions([func])
                    if 'dv_struct' in d_inputs:
                        dvsens = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                        self.tacs_assembler.evalDVSens(func, dvsens)

                        d_inputs['dv_struct'][:] += np.array(dvsens,dtype=float) * d_outputs['func_struct'][ifunc]

                    if 'x_struct0' in d_inputs:
                        xpt_sens = self.xpt_sens
                        xpt_sens_array = xpt_sens.getArray()
                        self.tacs_assembler.evalXptSens(func, xpt_sens)

                        d_inputs['x_struct0'][:] += np.array(xpt_sens_array,dtype=float) * d_outputs['func_struct'][ifunc]

                    if 'u_struct' in d_inputs:
                        prod = self.tacs_assembler.createVec()
                        self.tacs_assembler.evalSVSens(func,prod)
                        prod_array = prod.getArray()

                        d_inputs['u_struct'][:] += np.array(prod_array,dtype=float) * d_outputs['func_struct'][ifunc]

class TacsMass(om.ExplicitComponent):
    """
    Component to compute TACS mass
    """
    def initialize(self):
        self.options.declare('tacs_assembler', recordable=False)
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')

        self.options['distributed'] = True

        self.ans = None
        self.tacs_assembler = None

        self.mass = False

        self.check_partials = False

    def setup(self):

        self.tacs_assembler = self.options['tacs_assembler']
        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        #self.set_check_partial_options(wrt='*',directional=True)

        tacs_assembler = self.tacs_assembler

        # TACS part of setupk
        self.tacs_assembler = tacs_assembler
        ndv  = self.struct_objects[3]['ndv']

        self.xpt_sens = tacs_assembler.createNodeVec()

        # OpenMDAO part of setup
        self.add_input('dv_struct', shape=ndv,          desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', shape_by_conn=True, desc='structural node coordinates', tags=['mphys_coordinates'])

        self.add_output('mass', 0.0, desc = 'structural mass', tags=['mphys_result'])
        #self.declare_partials('mass',['dv_struct','x_struct0'])

    def _update_internal(self,inputs):
        self.tacs_assembler.setDesignVars(np.array(inputs['dv_struct'],dtype=TACS.dtype))

        xpts = self.tacs_assembler.createNodeVec()
        self.tacs_assembler.getNodes(xpts)
        xpts_array = xpts.getArray()
        xpts_array[:] = inputs['x_struct0']
        self.tacs_assembler.setNodes(xpts)

    def compute(self,inputs,outputs):
        if self.check_partials:
            self._update_internal(inputs)

        if 'mass' in outputs:
            func = functions.StructuralMass(self.tacs_assembler)
            outputs['mass'] = self.tacs_assembler.evalFunctions([func])

    def compute_jacvec_product(self,inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if self.check_partials:
                pass
            else:
                raise ValueError('TACS forward mode requested but not implemented')
        if mode == 'rev':
            if self.check_partials:
                self._update_internal(inputs)
            if 'mass' in d_outputs:
                func = functions.StructuralMass(self.tacs_assembler)
                if 'dv_struct' in d_inputs:
                    dvsens = np.zeros(d_inputs['dv_struct'].size,dtype=TACS.dtype)
                    self.tacs_assembler.evalDVSens(func, dvsens)

                    d_inputs['dv_struct'] += np.array(dvsens,dtype=float) * d_outputs['mass']

                if 'x_struct0' in d_inputs:
                    xpt_sens = self.xpt_sens
                    xpt_sens_array = xpt_sens.getArray()
                    self.tacs_assembler.evalXptSens(func, xpt_sens)
                    d_inputs['x_struct0'] += np.array(xpt_sens_array,dtype=float) * d_outputs['mass']


class PrescribedLoad(om.ExplicitComponent):
    """
    Prescribe a load to tacs
    """
    def initialize(self):
        self.options.declare('load_function', default = None, desc='function that prescribes the loads', recordable=False)
        self.options.declare('tacs_assembler', recordable=False)

        self.options['distributed'] = True

        self.ndof = 0

    def setup(self):
#        self.set_check_partial_options(wrt='*',directional=True)

        # TACS assembler setup
        tacs_assembler = self.options['tacs_assembler']

        # create some TACS vectors so we can see what size they are
        # TODO getting the node sizes should be easier than this...
        xpts  = tacs_assembler.createNodeVec()
        node_size = xpts.getArray().size

        tmp   = tacs_assembler.createVec()
        state_size = tmp.getArray().size
        self.ndof = int(state_size / ( node_size / 3 ))

        # OpenMDAO setup
        self.add_input('x_struct0', shape_by_conn=True, desc='structural node coordinates', tags=['mphys_coordinates'])
        self.add_output('f_struct', shape=state_size,   desc='structural load', tags=['mphys_coupling'])

        #self.declare_partials('f_struct','x_struct0')

    def compute(self,inputs,outputs):
        load_function = self.options['load_function']
        outputs['f_struct'] = load_function(inputs['x_struct0'],self.ndof)

class TacsGroup(om.Group):
    def initialize(self):
        self.options.declare('tacs_assembler', recordable=False)
        self.options.declare('solver_objects', recordable=False)
        self.options.declare('check_partials')
        self.options.declare('conduction', default=False)


    def setup(self):
        self.tacs_assembler = self.options['tacs_assembler']
        self.struct_objects = self.options['solver_objects']
        self.check_partials = self.options['check_partials']

        # check if we have a loading function
        solver_dict = self.struct_objects[3]

        if 'load_function' in solver_dict:
            self.prescribed_load = True
            self.add_subsystem('loads', PrescribedLoad(
                load_function=solver_dict['load_function'],
                tacs_assembler=self.tacs_assembler
            ), promotes_inputs=['x_struct0'], promotes_outputs=['f_struct'])

        if self.options['conduction']:
            self.add_subsystem('solver', TacsSolverConduction(
                tacs_assembler=self.tacs_assembler,
                struct_objects=self.struct_objects,
                check_partials=self.check_partials),
                promotes_inputs=['q_conduct', 'x_struct0'],
                promotes_outputs=['T_conduct']
            )
        else:
            self.add_subsystem('solver', TacsSolver(
                tacs_assembler=self.tacs_assembler,
                struct_objects=self.struct_objects,
                check_partials=self.check_partials),
                promotes_inputs=['f_struct', 'x_struct0', 'dv_struct'],
                promotes_outputs=['u_struct']
            )


class TACSFuncsGroup(om.Group):
    def initialize(self):
        self.options.declare('tacs_assembler', recordable=False)
        self.options.declare('solver_objects', recordable=False)
        self.options.declare('check_partials')

    def setup(self):
        self.tacs_assembler = self.options['tacs_assembler']
        self.struct_objects = self.options['solver_objects']
        self.check_partials = self.options['check_partials']

        self.add_subsystem('funcs', TacsFunctions(
            tacs_assembler=self.tacs_assembler,
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['x_struct0', 'u_struct','dv_struct'],
            promotes_outputs=['func_struct']
        )

        self.add_subsystem('mass', TacsMass(
            tacs_assembler=self.tacs_assembler,
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['x_struct0', 'dv_struct'],
            promotes_outputs=['mass'],
        )

    def configure(self):
        pass

class TacsBuilder(Builder):

    def __init__(self, options, check_partials=False, conduction=False):
        self.options = options
        self.check_partials = check_partials
        self.conduction = conduction

    def initialize(self, comm):
        solver_dict={}

        mesh = TACS.MeshLoader(comm)
        mesh.scanBDFFile(self.options['mesh_file'])

        ndof, ndv = self.options['add_elements'](mesh)

        tacs_assembler = mesh.createTACS(ndof)

        number_of_nodes = tacs_assembler.createNodeVec().getArray().size // 3

        if self.conduction:
            mat = tacs_assembler.createSchurMat()
        else:
            mat = tacs_assembler.createFEMat()

        pc = TACS.Pc(mat)

        subspace = 100
        restarts = 2
        gmres = TACS.KSM(mat, pc, subspace, restarts)

        solver_dict['ndv']    = ndv
        solver_dict['ndof']   = ndof
        solver_dict['number_of_nodes'] = number_of_nodes
        solver_dict['get_funcs'] = self.options['get_funcs']

        #use the supplied function to get the surface points and mapping
        if self.conduction:
            solver_dict['surface_nodes'], solver_dict['mapping'] = self.options['get_surface'](tacs_assembler)

        if 'f5_writer' in self.options.keys():
            solver_dict['f5_writer'] = self.options['f5_writer']

        # check if the user provided a load function
        if 'load_function' in self.options.keys():
            solver_dict['load_function'] = self.options['load_function']

        self.solver_dict=solver_dict
        self.tacs_assembler = tacs_assembler
        self.solver_objects = [mat, pc, gmres, solver_dict]

    def get_coupling_group_subsystem(self, **kwargs):
        return TacsGroup(tacs_assembler=self.tacs_assembler,
                         solver_objects=self.solver_objects,
                         check_partials=self.check_partials,
                         **kwargs)

    def get_mesh_coordinate_subsystem(self):
        return TacsMesh(tacs_assembler=self.tacs_assembler)

    def get_post_coupling_subsystem(self):
        return TACSFuncsGroup(
            tacs_assembler=self.tacs_assembler,
            solver_objects=self.solver_objects,
            check_partials=self.check_partials
        )

    def get_ndof(self):
        return self.solver_dict['ndof']

    def get_number_of_nodes(self):
        return self.solver_dict['number_of_nodes']

    def get_ndv(self):
        return self.solver_dict['ndv']
