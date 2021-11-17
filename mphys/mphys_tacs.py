import numpy as np

import openmdao.api as om
from mphys.builder import Builder
from tacs import pyTACS


class TacsMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates with TACS
    """

    def initialize(self):
        self.options.declare('fea_solver', default=None, desc='the pytacs object itself', recordable=False)

    def setup(self):
        fea_solver = self.options['fea_solver']
        xpts = fea_solver.getOrigCoordinates()
        self.add_output('x_struct0', distributed=True, val=xpts, shape=xpts.size,
                        desc='structural node coordinates', tags=['mphys_coordinates'])


class TacsSolver(om.ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - The TACS steady residual is R = K * u_s - f_s = 0

    """

    def initialize(self):
        self.options.declare('fea_solver', recordable=False)
        self.options.declare('check_partials')

        self.fea_solver = None

        self.transposed = False
        self.check_partials = False

        self.old_dvs = None
        self.old_xs = None

    def setup(self):
        self.check_partials = self.options['check_partials']

        self.fea_solver = self.options['fea_solver']

        # OpenMDAO setup
        ndv = self.fea_solver.getTotalNumDesignVars()
        self.ndof = self.fea_solver.getVarsPerNode()
        state_size = self.fea_solver.getNumOwnedNodes() * self.ndof

        # inputs
        self.add_input('dv_struct', distributed=False, shape=ndv,
                       desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=True, shape_by_conn=True,
                       desc='structural node coordinates', tags=['mphys_coordinates'])
        '''
        self.add_input('rhs',  distributed=True, shape=state_size, val = 0.0,
                       desc='structural load vector', tags=['mphys_coupling'])
        '''

        # outputs
        # its important that we set this to zero since this displacement value is used for the first iteration of the aero
        self.add_output('states', distributed=True, shape=state_size, val=np.zeros(state_size),
                        desc='structural state vector', tags=['mphys_coupling'])

    def _need_update(self, inputs):
        update = False

        if self.old_dvs is None:
            self.old_dvs = inputs['dv_struct'].copy()
            update = True

        for dv, dv_old in zip(inputs['dv_struct'], self.old_dvs):
            if np.abs(dv - dv_old) > 0.:  # 1e-7:
                self.old_dvs = inputs['dv_struct'].copy()
                update = True

        if self.old_xs is None:
            self.old_xs = inputs['x_struct0'].copy()
            update = True

        for xs, xs_old in zip(inputs['x_struct0'], self.old_xs):
            if np.abs(xs - xs_old) > 0.:  # 1e-7:
                self.old_xs = inputs['x_struct0'].copy()
                update = True

        return update

    def _update_internal(self, inputs, outputs=None):
        if self._need_update(inputs):
            self.sp.setDesignVars(inputs['dv_struct'])
            self.sp.setCoordinates(inputs['x_struct0'])
        if outputs is not None:
            self.sp.setVariables(outputs['states'])
        self.sp._setProblemVars()

    def apply_nonlinear(self, inputs, outputs, residuals):
        self._update_internal(inputs, outputs)
        self.sp.getResidual(res=residuals['states'])  # ,
        # Fext=inputs['rhs'])

    def solve_nonlinear(self, inputs, outputs):
        self._update_internal(inputs)
        self.sp.solve()  # Fext=inputs['rhs'])
        self.sp.getVariables(states=outputs['states'])
        self.sp.writeSolution()

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            if self.check_partials:
                print('solver fwd')
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            self.sp.solveAdjoint(d_outputs['states'], d_residuals['states'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        self._update_internal(inputs, outputs)
        if mode == 'fwd':
            if not self.check_partials:
                raise ValueError('TACS forward mode requested but not implemented')

        if mode == 'rev':
            if 'states' in d_residuals:

                if 'states' in d_outputs:
                    self.sp.addTransposeJacVecProduct(d_residuals['states'],
                                                      d_outputs['states'])

                if 'rhs' in d_inputs:
                    d_inputs['rhs'] -= d_residuals['states']

                if 'x_struct0' in d_inputs:
                    self.sp.addAdjointResXptSensProducts([d_residuals['states']], [d_inputs['x_struct0']], scale=1.0)

                if 'dv_struct' in d_inputs:
                    self.sp.addAdjointResProducts([d_residuals['states']], [d_inputs['dv_struct']], scale=1.0)

    def _design_vector_changed(self, x):
        if self.x_save is None:
            self.x_save = x.copy()
            return True
        elif not np.allclose(x, self.x_save, rtol=1e-10, atol=1e-10):
            self.x_save = x.copy()
            return True
        else:
            return False

    def set_sp(self, sp):
        self.sp = sp


class TacsFunctions(om.ExplicitComponent):
    """
    Component to compute TACS functions
    """

    def initialize(self):
        self.options.declare('fea_solver', recordable=False)
        self.options.declare('check_partials')

        self.fea_solver = None

        self.check_partials = False

    def setup(self):
        self.fea_solver = self.options['fea_solver']
        self.check_partials = self.options['check_partials']

        # TACS part of setup
        self.ndv = ndv = self.fea_solver.getTotalNumDesignVars()

        # OpenMDAO part of setup
        # TODO move the dv_struct to an external call where we add the DVs
        self.add_input('dv_struct', distributed=False, shape=ndv, desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='structural node coordinates',
                       tags=['mphys_coordinates'])
        self.add_input('states', distributed=True, shape_by_conn=True, desc='structural state vector',
                       tags=['mphys_coupling'])

    def mphys_set_sp(self, sp):
        # this is the external function to set the sp to this component
        self.sp = sp

        # Add eval funcs as outputs
        for func_name in self.sp.functionList:
            self.add_output(func_name, distributed=False, shape=1, tags=["mphys_result"])

    def _update_internal(self, inputs):
        self.sp.setDesignVars(inputs['dv_struct'])
        self.sp.setCoordinates(inputs['x_struct0'])
        self.sp.setVariables(inputs['states'])

    def compute(self, inputs, outputs):
        if self.check_partials:
            self._update_internal(inputs)

        # Evaluate functions
        funcs = {}
        self.sp.evalFunctions(funcs)
        for key in funcs:
            # Remove struct problem name from key
            func_name = key.replace(self.sp.name + '_', '')
            outputs[func_name] = funcs[key]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if not self.check_partials:
                raise ValueError('TACS forward mode requested but not implemented')
        if mode == 'rev':
            # always update internal because same tacs object could be used by multiple scenarios
            # and we need to load this scenario's state back into TACS before doing derivatives
            self._update_internal(inputs)

            for func_name in d_outputs:
                proc_contribution = d_outputs[func_name][:]
                d_func = self.comm.allreduce(proc_contribution) / self.comm.size

                if 'dv_struct' in d_inputs:
                    self.sp.addDVSens([func_name], [d_inputs['dv_struct']], scale=d_func)

                if 'x_struct0' in d_inputs:
                    self.sp.addXptSens([func_name], [d_inputs['x_struct0']], scale=d_func)

                if 'states' in d_inputs:
                    sv_sens = np.zeros_like(d_inputs['states'])
                    self.sp.addSVSens([func_name], [sv_sens])
                    d_inputs['states'][:] += sv_sens * d_func


class TacsGroup(om.Group):
    def initialize(self):
        self.options.declare('fea_solver', recordable=False)
        self.options.declare('check_partials')
        self.options.declare('conduction', default=False)

    def setup(self):
        self.fea_solver = self.options['fea_solver']
        self.check_partials = self.options['check_partials']

        # Promote state variables/rhs with physics-specific tag that MPhys expects
        if self.options['conduction']:
            promotes_inputs = ['x_struct0', 'dv_struct']  # [('rhs', 'q_conduct'), 'x_struct0', 'dv_struct']
            promotes_outputs = [('states', 'T_conduct')]
        else:
            promotes_inputs = ['x_struct0', 'dv_struct']  # [('rhs', 'f_struct'), 'x_struct0', 'dv_struct']
            promotes_outputs = [('states', 'u_struct')]

        self.add_subsystem('solver', TacsSolver(
            fea_solver=self.fea_solver,
            check_partials=self.check_partials),
                           promotes_inputs=promotes_inputs,
                           promotes_outputs=promotes_outputs)

        # Default structural problem
        sp = self.fea_solver.createStaticProblem(name='wing')
        self.mphys_set_sp(sp)

    def mphys_set_sp(self, sp):
        """
        Allows user to attach custom structural problem. The problem can include
        fixed structural loads that may have been added to the problem by the user.
        """
        # set the struct problem
        self.sp = sp
        self.solver.set_sp(sp)


class TACSFuncsGroup(om.Group):
    def initialize(self):
        self.options.declare('fea_solver', recordable=False)
        self.options.declare('check_partials')
        self.options.declare('conduction', default=False)

    def setup(self):
        self.fea_solver = self.options['fea_solver']
        self.check_partials = self.options['check_partials']

        # Promote state variables/rhs with physics-specific tag that MPhys expects
        if self.options['conduction']:
            promotes_inputs = [('states', 'T_conduct'), 'x_struct0', 'dv_struct']
        else:
            promotes_inputs = [('states', 'u_struct'), 'x_struct0', 'dv_struct']

        self.add_subsystem('funcs', TacsFunctions(
            fea_solver=self.fea_solver,
            check_partials=self.check_partials),
                           promotes_inputs=promotes_inputs,
                           promotes_outputs=['*']
                           )

        # Default structural problem
        sp = self.fea_solver.createStaticProblem(name='wing')
        self.mphys_set_sp(sp)

    def mphys_set_sp(self, sp):
        # this is the external function to set the sp to this component
        self.sp = sp
        self.funcs.mphys_set_sp(sp)


class TacsBuilder(Builder):

    def __init__(self, options, check_partials=False, conduction=False):
        self.options = options
        self.check_partials = check_partials
        self.conduction = conduction

    def initialize(self, comm):
        bdf_file = self.options.pop('mesh_file')

        if 'element_callback' in self.options:
            element_callback = self.options.pop('element_callback')
        else:
            element_callback = None

        self.fea_solver = pyTACS(bdf_file, options=self.options, comm=comm)

        # Set up elements and TACS assembler
        self.fea_solver.createTACSAssembler(element_callback)

    def get_coupling_group_subsystem(self):
        return TacsGroup(fea_solver=self.fea_solver,
                         check_partials=self.check_partials)

    def get_mesh_coordinate_subsystem(self):
        return TacsMesh(fea_solver=self.fea_solver)

    def get_post_coupling_subsystem(self):
        return TACSFuncsGroup(
            fea_solver=self.fea_solver,
            check_partials=self.check_partials,
            conduction=self.conduction
        )

    def get_ndof(self):
        return self.fea_solver.getVarsPerNode()

    def get_number_of_nodes(self):
        return self.fea_solver.getNumOwnedNodes()

    def get_ndv(self):
        return self.fea_solver.getTotalNumDesignVars()

    def get_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.fea_solver.assembler

    def get_fea_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.fea_solver
