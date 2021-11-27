import numpy as np

import openmdao.api as om
from mphys.builder import Builder
from tacs import pyTACS
from openmdao.utils.mpi import MPI

class TacsMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates with TACS
    """

    def initialize(self):
        self.options.declare('fea_solver', default=None, desc='the pytacs object itself', recordable=False)

    def setup(self):
        fea_solver = self.options['fea_solver']
        xpts = fea_solver.getOrigNodes()
        self.add_output('x_struct0', distributed=True, val=xpts, shape=xpts.size,
                        desc='structural node coordinates', tags=['mphys_coordinates'])

class TacsDVComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('fea_solver', default=None, desc='the pytacs object itself', recordable=False)
        self.options.declare('initial_dv_vals', default=None, desc='initial values for global design variable vector')

    def setup(self):
        self.fea_solver = self.options['fea_solver']
        self.src_indices = self.get_dv_src_indices()
        vals = self.options['initial_dv_vals']
        ndv = self.fea_solver.getNumDesignVars()
        self.add_input('dv_struct', val=vals, distributed=False, tags=['mphys_input'])
        self.add_output('dv_struct_distributed',shape=ndv, distributed=True, tags=['mphys_coupling'])

    def compute(self, inputs, outputs):
        outputs['dv_struct_distributed'] = inputs['dv_struct'][self.src_indices]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'dv_struct_distributed' in d_outputs:
                d_outputs['dv_struct_distributed'] += d_inputs['dv_struct'][self.src_indices]
        else: # mode == 'rev'
            if 'dv_struct' in d_inputs:
                if MPI is not None and self.comm.size > 1:
                    deriv = np.zeros_like(d_inputs['dv_struct'])
                    deriv[self.src_indices] = d_outputs['dv_struct_distributed']
                    deriv_sum = np.zeros_like(d_inputs['dv_struct'])
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_inputs['dv_struct'] += deriv_sum
                else:
                    d_inputs['dv_struct'] += d_outputs['dv_struct_distributed']

    def get_dv_src_indices(self):
        """
        Method to get src_indices on each processor
        for tacs distributed design variable vec
        """
        if MPI is not None and self.comm.size > 1:
            local_ndvs = self.fea_solver.getNumDesignVars()
            all_proc_ndvs = self.comm.gather(local_ndvs, root=0)
            all_proc_indices = []
            if self.comm.rank == 0:
                tot_ndvs = 0
                for proc_i in range(self.comm.size):
                    local_ndvs = all_proc_ndvs[proc_i]
                    proc_indices = np.arange(tot_ndvs, tot_ndvs + local_ndvs)
                    all_proc_indices.append(proc_indices)
                    tot_ndvs += local_ndvs
            local_dv_indices = self.comm.scatter(all_proc_indices, root=0)
            return local_dv_indices
        else:
            ndvs = len(self.options['initial_dv_vals'])
            all_dv_indices = np.arange(ndvs)
            return all_dv_indices

class TacsSolver(om.ImplicitComponent):
    """
    Component to perform TACS steady analysis

    Assumptions:
        - The TACS steady residual is R = K * u_s - f_s = 0

    """

    def initialize(self):
        self.options.declare('fea_solver', recordable=False)
        self.options.declare('check_partials')
        self.options.declare('coupled', default=False)

        self.fea_solver = None

        self.transposed = False
        self.check_partials = False

        self.old_dvs = None
        self.old_xs = None

    def setup(self):
        self.check_partials = self.options['check_partials']
        self.fea_solver = self.options['fea_solver']
        self.coupled = self.options['coupled']

        # OpenMDAO setup
        local_ndvs = self.fea_solver.getNumDesignVars()
        self.ndof = self.fea_solver.getVarsPerNode()
        state_size = self.fea_solver.getNumOwnedNodes() * self.ndof

        # inputs
        self.add_input('dv_struct_distributed', distributed=True, shape=local_ndvs,
                       desc='tacs distributed design variables', tags=['mphys_coupling'])
        self.add_input('x_struct0', distributed=True, shape_by_conn=True,
                       desc='distributed structural node coordinates', tags=['mphys_coordinates'])
        if self.coupled:
            self.add_input('rhs',  distributed=True, shape=state_size, val=0.0,
                           desc='coupling load vector', tags=['mphys_coupling'])

        # outputs
        # its important that we set this to zero since this displacement value is used for the first iteration of the aero
        self.add_output('states', distributed=True, shape=state_size, val=np.zeros(state_size),
                        desc='structural state vector', tags=['mphys_coupling'])

    def _need_update(self, inputs):
        update = True

        if self.old_dvs is None:
            self.old_dvs = inputs['dv_struct_distributed'].copy()
            update = True

        for dv, dv_old in zip(inputs['dv_struct_distributed'], self.old_dvs):
            if np.abs(dv - dv_old) > 0.:  # 1e-7:
                self.old_dvs = inputs['dv_struct_distributed'].copy()
                update = True

        if self.old_xs is None:
            self.old_xs = inputs['x_struct0'].copy()
            update = True

        for xs, xs_old in zip(inputs['x_struct0'], self.old_xs):
            if np.abs(xs - xs_old) > 0.:  # 1e-7:
                self.old_xs = inputs['x_struct0'].copy()
                update = True
        tmp = [update]
        # Perform all reduce to check if any other procs came back True
        update = self.comm.allreduce(tmp)[0]
        return update

    def _update_internal(self, inputs, outputs=None):
        if self._need_update(inputs):
            self.sp.setDesignVars(inputs['dv_struct_distributed'])
            self.sp.setNodes(inputs['x_struct0'])
        if outputs is not None:
            self.sp.setVariables(outputs['states'])
        self.sp._updateAssemblerVars()

    def apply_nonlinear(self, inputs, outputs, residuals):
        self._update_internal(inputs, outputs)

        if self.coupled:
            Fext = inputs['rhs']
        else:
            Fext = None

        self.sp.getResidual(res=residuals['states'], Fext=Fext)

    def solve_nonlinear(self, inputs, outputs):
        self._update_internal(inputs)

        if self.coupled:
            Fext = inputs['rhs']
        else:
            Fext = None

        self.sp.solve(Fext=Fext)
        self.sp.getVariables(states=outputs['states'])

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
                    array_w_bcs = d_residuals['states'].copy()
                    self.fea_solver.applyBCsToVec(array_w_bcs)
                    d_inputs['rhs'] -= array_w_bcs

                if 'x_struct0' in d_inputs:
                    self.sp.addAdjointResXptSensProducts([d_residuals['states']], [d_inputs['x_struct0']], scale=1.0)

                if 'dv_struct_distributed' in d_inputs:
                    self.sp.addAdjointResProducts([d_residuals['states']], [d_inputs['dv_struct_distributed']], scale=1.0)

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
        self.options.declare("write_solution", default=True)

        self.fea_solver = None

        self.check_partials = False

    def setup(self):
        self.fea_solver = self.options['fea_solver']
        self.check_partials = self.options['check_partials']
        self.write_solution = self.options["write_solution"]
        self.solution_counter = 0

        # TACS part of setup
        local_ndvs = self.fea_solver.getNumDesignVars()

        # OpenMDAO part of setup
        self.add_input('dv_struct_distributed', distributed=True, shape=local_ndvs,
                       desc='tacs design variables', tags=['mphys_coupling'])
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
        self.sp.setDesignVars(inputs['dv_struct_distributed'])
        self.sp.setNodes(inputs['x_struct0'])
        states_w_bcs = inputs['states'].copy()
        self.fea_solver.applyBCsToVec(states_w_bcs)
        self.sp.setVariables(states_w_bcs)

    def compute(self, inputs, outputs):
        self._update_internal(inputs)

        # Evaluate functions
        funcs = {}
        self.sp.evalFunctions(funcs)
        for key in funcs:
            # Remove struct problem name from key
            func_name = key.replace(self.sp.name + '_', '')
            outputs[func_name] = funcs[key]

        if self.write_solution:
            # write the solution files.
            self.sp.writeSolution(number=self.solution_counter)
            self.solution_counter += 1

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if not self.check_partials:
                raise ValueError('TACS forward mode requested but not implemented')
        if mode == 'rev':
            # always update internal because same tacs object could be used by multiple scenarios
            # and we need to load this scenario's state back into TACS before doing derivatives
            self._update_internal(inputs)

            for func_name in d_outputs:
                d_func = d_outputs[func_name][0]

                if 'dv_struct_distributed' in d_inputs:
                    self.sp.addDVSens([func_name], [d_inputs['dv_struct_distributed']], scale=d_func)

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
        self.options.declare('coupled', default=False)

    def setup(self):
        self.fea_solver = self.options['fea_solver']
        self.check_partials = self.options['check_partials']
        self.coupled = self.options['coupled']

        # Promote state variables/rhs with physics-specific tag that MPhys expects
        if self.options['conduction']:
            promotes_inputs = ['x_struct0', 'dv_struct_distributed']
            promotes_outputs = [('states', 'T_conduct')]
            if self.coupled:
                promotes_inputs.append(('rhs', 'q_conduct'))
        else:
            promotes_inputs = ['x_struct0', 'dv_struct_distributed']
            promotes_outputs = [('states', 'u_struct')]
            if self.coupled:
                promotes_inputs.append(('rhs', 'f_struct'))

        self.add_subsystem('solver',
                           TacsSolver(fea_solver=self.fea_solver, check_partials=self.check_partials,
                                      coupled=self.coupled),
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

        # Promote state variables with physics-specific tag that MPhys expects
        if self.options['conduction']:
            promotes_inputs = [('states', 'T_conduct'), 'x_struct0', 'dv_struct_distributed']
        else:
            promotes_inputs = [('states', 'u_struct'), 'x_struct0', 'dv_struct_distributed']

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

    def __init__(self, options, check_partials=False, conduction=False, coupled=False):
        self.options = options
        self.check_partials = check_partials
        self.conduction = conduction
        # Flag to turn on coupling variables
        self.coupled = coupled

    def initialize(self, comm):
        bdf_file = self.options.pop('mesh_file')

        if 'element_callback' in self.options:
            element_callback = self.options.pop('element_callback')
        else:
            element_callback = None

        self.fea_solver = pyTACS(bdf_file, options=self.options, comm=comm)
        self.comm = comm

        # Set up elements and TACS assembler
        self.fea_solver.initialize(element_callback)

    def get_coupling_group_subsystem(self):
        return TacsGroup(fea_solver=self.fea_solver,
                         check_partials=self.check_partials,
                         coupled=self.coupled)

    def get_mesh_coordinate_subsystem(self):
        return TacsMesh(fea_solver=self.fea_solver)

    def get_pre_coupling_subsystem(self):
        initial_dvs = self.get_initial_dvs()
        return TacsDVComp(fea_solver=self.fea_solver, initial_dv_vals=initial_dvs)

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

    def get_initial_dvs(self):
        """
        Get an array holding all dvs values that have been added to TACS
        """
        if MPI is not None and self.comm.size > 1:
            # Get DVs locally owned by this processor
            local_dvs = self.fea_solver.getOrigDesignVars()
            local_dvs = local_dvs.astype(float)
            # Size of design variable on this processor
            local_ndvs = self.fea_solver.getNumDesignVars()
            # Size of design variable vector on each processor
            dv_sizes = self.comm.allgather(local_ndvs)
            # Offsets for global design variable vector
            offsets = np.zeros(self.comm.size, dtype=int)
            offsets[1:] = np.cumsum(dv_sizes)[:-1]
            # Gather the portions of the design variable array distributed across each processor
            tot_ndvs = self.fea_solver.getTotalNumDesignVars()
            global_dvs = np.zeros(tot_ndvs, dtype=local_dvs.dtype)
            self.comm.Allgatherv(local_dvs, [global_dvs, dv_sizes, offsets, MPI.DOUBLE])
            # return the global dv array
            return global_dvs
        else:
            return self.fea_solver.getOrigDesignVars()

    def get_ndv(self):
        return self.fea_solver.getTotalNumDesignVars()

    def get_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.fea_solver.assembler

    def get_fea_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.fea_solver
