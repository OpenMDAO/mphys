import numpy as np
from mpi4py import MPI

from openmdao.api import ImplicitComponent, ParallelGroup
from tacs import functions


class StructuralGroup(ParallelGroup):
    def initialize(self):
        self.options.declare('struct_comp', desc='Structural solver component')
        self.options.declare('nprocs',default=self.comm.Get_size(),desc='number of structural processors')

    def setup(self):
        struct_comp = self.options['struct_comp']
        nprocs      = self.options['nprocs']

        self.add_subsystem('struct_comp', struct_comp, promotes_inputs ='*'
                                                     , promotes_outputs='*'
                                                     , max_procs=nprocs)


class TacsSolver(ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - Mesh will be loaded from bdf file
        - user will provide an add_elements function that assigns tacs elements to the mesh and returns the ndof per node

    """
    def initialize(self):
        self.options.declare('tac_setup', default = None, desc='Function to setup tacs')

        self.options['distributed'] = True

        self.tacs = None
        self.pc = None


        self.res = None
        self.ans = None
        self.svsens = None
        self.struct_rhs = None
        self.psi_S = None
        self.x_save = None

    def setup(self):

        # TACS assembler setup
        tacs_setup = self.options['tacs_setup']
        tacs, pc, gmres, ndv = tacs_setup(self.comm)

        self.tacs = tacs
        self.pc = pc
        self.gmres = gmres

        # create some TACS bvecs that will be needed later
        self.res        = tacs.createVec()
        self.ans        = tacs.createVec()
        self.svsens     = tacs.createVec()
        self.struct_rhs = tacs.createVec()
        self.psi_S      = tacs.createVec()

        # OpenMDAO setup
        xpts = tacs.createNodeVec()
        node_size = xpts.getArray().size

        # inputs
        self.add_input('struct_dv', shape=ndv       , desc='tacs design variables')
        self.add_input('x_s',       shape=node_size , desc='structural node coordinates')
        self.add_input('f_s',       shape=state_size, desc='structural load vector')

        # outputs
        self.add_output('u_s',      shape=state_size, desc='structural state vector')

        # partials
        self.declare_partials('u_s',['x','x_s','f_s'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        The TACS steady residual is R = K * u - f = 0
        """
        tacs = self.tacs
        res  = self.res
        ans  = self.ans

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
        force  = self.res # using res bvec as force so we don't have to allocate more memory
        ans    = self.ans

        # if the design variables changed, update the stiffness matrix
        if self._design_vector_changed(inputs['x']):
            tacs.setDesignVars(inputs['x'])
            alpha = 1.0
            beta  = 0.0
            gamma = 0.0
            tacs.assembleJacobian(alpha,beta,gamma,res,mat)
            pc.factor()

        # solve the linear system
        force_array = force.getArray()
        force_array[:] = inputs['f_s']

        pc.applyFactor(force, ans)
        ans_array = ans.getArray()
        outputs['u_s'] = ans_array[:]

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
       if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_s' in d_residuals:
                if 'u_s' in d_outputs:

                if 'f_s' in d_inputs:

    def _design_vector_changed(self,x):
        if self.x_save is None:
            self.x_save = x
            return True
        elif np.allclose(x,self.x_save,rtol=1e-10,atol=1e-13)
            self.x_save = x
            return True
        else:
            return False


class TACSfunctions(ExplicitComponent):
    def initialize(self):
        self.options.declare('tacs_func_setup', default = None, desc = 'tacs assembler')

        self.ans = None

    def setup(self):
        tacs_func_setup = self.options['tacs_func_setup']
        tacs, func_list, ndv = tacs_func_setup(self.comm)

        self.tacs = tacs

        self.ans = tacs.createVec()

        self.add_input('stuct_dv',  shape=ndv,            desc='tacs design variables')
        self.add_input('x_s',       shape=xpts_shape,     desc='structural node coordinates')
        self.add_input('u_s',       shape=state_shape,    desc='structural state vector')

        # Remove the mass function from the func list if it is there
        # since it isn't dependent on the structural state
        self.mass = True
        func_no_mass = []
        for i,func in enumerate(func_list)):
            if isinstance(func,functions.StructuralMass):
                if not self.mass:
                    self.add_output('mass', 0.0, desc = 'structural mass')
                    self.mass = True
            else:
                func_no_mass.append(func)

        self.func_list = func_no_mass
        self.add_output('f_struct', shape=len(self.func_list), desc='structural function values')

        # declare the partials
        self.declare_partials('f_struct',['x_s','u_s'])
        self.declare_partials('mass',['x_s'])

    def compute(self,inputs,outputs):

        self.tacs.setVariables(self.ans)

        # the inputs are assumed to be passed from tacs object
        outputs['f_struct'] = self.tacs.evalFunctions(self.func_list)

        if 'mass' in outputs:
            funclist = [functions.structuralMass()]
            outputs['mass'] = self.tacs.evalFunctions(funclist)

    def compute_jacvec_product(inputs, d_inputs, d_outputs, mode):
