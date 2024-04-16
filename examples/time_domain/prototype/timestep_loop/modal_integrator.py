import numpy as np
import openmdao.api as om

from modal_comps import ModalDisplacements, ModalForces, ModalStep, HarmonicForcer

class ModalIntegrator(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes',default=1)
        self.options.declare('nsteps',default=10)
        self.options.declare('dt',default=1.0)
        self.options.declare('root_name',default='')

    def setup(self):
        nmodes = self.options['nmodes']
        self.add_input('z0',shape=nmodes)
        self.add_input('m',shape=nmodes)
        self.add_input('c',shape=nmodes)
        self.add_input('k',shape=nmodes)
        self.add_input('amp')
        self.add_input('freq')
        self.add_output('z_end',desc = 'final displacement')

        self._setup_step_problem()

        # temporary storage of in memory long term option is async file io
        self.z = np.zeros((self.options['nsteps']+1,nmodes))

    def _setup_step_problem(self):
        """
        OpenMDAO problem solved at each time step
        """
        nmodes    = self.options['nmodes']
        dt        = self.options['dt']
        root_name = self.options['root_name']

        self.problem = om.Problem()
        model = self.problem.model

        indeps = om.IndepVarComp()
        indeps.add_output('m',np.zeros(nmodes))
        indeps.add_output('c',np.zeros(nmodes))
        indeps.add_output('k',np.zeros(nmodes))
        indeps.add_output('amp')
        indeps.add_output('freq')
        indeps.add_output('time')

        indeps.add_output('znm1',np.zeros(nmodes))
        indeps.add_output('znm2',np.zeros(nmodes))
        indeps.add_output('znm3',np.zeros(nmodes))
        indeps.add_output('znm4',np.zeros(nmodes))

        model.add_subsystem('indeps',indeps)
        model.add_subsystem('modal_forces',ModalForces(nmodes=nmodes,root_name=root_name))
        model.add_subsystem('modal_solver',ModalStep(nmodes=nmodes,dt=dt))
        model.add_subsystem('modal_disps' ,ModalDisplacements(nmodes=nmodes,root_name=root_name))
        model.add_subsystem('forcer',HarmonicForcer(root_name=root_name))

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.LinearBlockGS()

        # coupling loop
        model.connect('modal_forces.mf','modal_solver.f')
        model.connect('modal_solver.zn','modal_disps.md')
        model.connect('modal_disps.dx','forcer.dx')
        model.connect('forcer.f','modal_forces.f')

        # modal properties
        model.connect('indeps.m','modal_solver.m')
        model.connect('indeps.c','modal_solver.c')
        model.connect('indeps.k','modal_solver.k')

        # forcer
        model.connect('indeps.amp','forcer.amp')
        model.connect('indeps.freq','forcer.freq')
        model.connect('indeps.time','forcer.time')

        # time backplanes
        model.connect('indeps.znm1','modal_solver.znm1')
        model.connect('indeps.znm2','modal_solver.znm2')
        model.connect('indeps.znm3','modal_solver.znm3')
        model.connect('indeps.znm4','modal_solver.znm4')

        self.problem.setup()

    def compute(self,inputs,outputs):

        # set design variables
        self._set_indeps(inputs)

        # start from step 1 to leave initial conditions as 0
        for step in range(1,self.options['nsteps']+1):
            self.problem['indeps.time'] = step*self.options['dt']
            self._setup_step_backplanes(step,inputs)
            self.problem.run_model()
            self._store_step_output(step)

        # compute integrator function of interest
        outputs['z_end'] = self.z[-1,0]

    def _set_indeps(self,inputs):
        self.problem['indeps.m'] = inputs['m']
        self.problem['indeps.c'] = inputs['c']
        self.problem['indeps.k'] = inputs['k']
        self.problem['indeps.amp'] = inputs['amp']
        self.problem['indeps.freq'] = inputs['freq']

    def _setup_step_backplanes(self,step,inputs):
        for backplane in range(1,5):
            prior_step = step - backplane
            if prior_step > 0:
                self.problem[f'indeps.znm{backplane}'] = self.z[prior_step,:]
            else:
                self.problem[f'indeps.znm{backplane}'] = inputs['z0']

    def _store_step_output(self,step):
        self.z[step,:] = self.problem['modal_solver.zn']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            self._forward_linearized_loop(inputs,d_inputs,d_outputs)
        if mode=='rev':
            self._reverse_linearized_loop(inputs,d_inputs,d_outputs)

    def _forward_linearized_loop(self,inputs,d_inputs,d_outputs):
        """
        d{}d{} is a total derivative
        p{}p{} is a partial derivative

        df/dx = pf/px + sum_{time_steps}  pf/pz * dz/dx
              = pf/px + sum_{time_steps}  pf/pz * ( pzpx + sum_{backplanes} pz/pz_{backplane} dz_{backplane}/dx )
        """
        dfdx = {}
        dznm_dx = {}
        for var in d_inputs.keys():
            if var == 'amp' or var=='freq':
                dfdx[var] = np.zeros(1)
                dznm_dx[var] =np.zeros((5,self.options['nmodes'],1))
            else:
                dfdx[var] = np.zeros(self.options['nmodes'])
                dznm_dx[var] =np.zeros((5,self.options['nmodes'],self.options['nmodes']))

        for step in range(1,self.options['nsteps']+1):

            pfpz = np.array([1.0,0.0]) if step == self.options['nsteps'] else np.zeros(2)

            self._set_state(step,inputs)
            self.problem.run_model() # if ALL the states are set I think this is unnecessary

            for var in d_inputs.keys():
                seed = {}
                wrt = ['indeps.znm1','indeps.znm2','indeps.znm3','indeps.znm4']
                if var == 'amp' or var == 'freq':
                    var_length = 1
                else:
                    var_length = 2
                if var != 'z0':
                    wrt.append('indeps.'+var)
                for var_index in range(var_length):
                    if var != 'z0':
                        if var == 'amp' or var == 'freq':
                            seed['indeps.'+var] = 1.0
                        else:
                            seed['indeps.'+var] = np.zeros(self.options['nmodes'])
                            seed['indeps.'+var][var_index] = 1.0

                    for j in range(1,5):
                        if step - j > 0:
                            seed[f'indeps.znm{j}'] = dznm_dx[var][j,:,var_index]
                        elif var=='z0':
                            seed[f'indeps.znm{j}'] =  np.zeros(self.options['nmodes'])
                            seed[f'indeps.znm{j}'][var_index] = 1.0
                        else:
                            seed[f'indeps.znm{j}'] = np.zeros(self.options['nmodes'])

                    # semi-totals
                    jac_vecs = self.problem.compute_jacvec_product(of=['modal_solver.zn'],wrt=wrt,mode='fwd',seed=seed)

                    dfdx[var][var_index] += pfpz.dot(jac_vecs['modal_solver.zn'])

                    dznm_dx[var][0,:,var_index] = jac_vecs['modal_solver.zn'].reshape(dznm_dx[var][0,:,var_index].shape)

                # shuffle the backplanes of dz/dx
                for j in range(4,0,-1):
                    dznm_dx[var][j,:,:] = dznm_dx[var][j-1,:,:]

        if 'z_end' in d_outputs:
            for var in d_inputs:
                d_outputs['z_end'] += dfdx[var].dot(d_inputs[var])

    def _reverse_linearized_loop(self,inputs,d_inputs,d_outputs):
        """
        d{}d{} is a total derivative
        p{}p{} is a partial derivative
        """
        dfdx = {}
        for var in d_inputs.keys():
            if var == 'amp' or var=='freq':
                dfdx[var] = np.zeros(1)
            else:
                dfdx[var] = np.zeros((self.options['nmodes'],1))

        lamb = np.zeros((self.options['nmodes'],5))
        for step in range(self.options['nsteps'],0,-1):
            pfpz = np.array([1.0,0.0]) if step == self.options['nsteps'] else np.zeros(2)

            self._set_state(step,inputs)
            self.problem.run_model() # if ALL the states are set I think this is unnecessary

            # compute the current adjoint
            psi = pfpz + lamb[:,1]

            # jacobian vector products
            jac_vecs = self.problem.compute_jacvec_product(of=['modal_solver.zn'],wrt=['indeps.m',
                                                                                       'indeps.c',
                                                                                       'indeps.k',
                                                                                       'indeps.amp',
                                                                                       'indeps.freq',
                                                                                       'indeps.znm1',
                                                                                       'indeps.znm2',
                                                                                       'indeps.znm3',
                                                                                       'indeps.znm4'],mode='rev',seed={'modal_solver.zn':psi})

            # add this step's contribution to the derivatives
            for var in d_inputs.keys():
                if var != 'z0':
                    dfdx[var] += jac_vecs['indeps.'+var].reshape(dfdx[var].shape)

                for j in range(1,5):
                    if var=='z0' and step - j < 1:
                        dfdx[var] += jac_vecs[f'indeps.znm{j}'].reshape(dfdx[var].shape)

            # shuffle the forwardplanes of adjoint variables and add this step's contributions
            for j in range(1,5):
                if j < 4:
                    lamb[:,j] = lamb[:,j+1]
                else:
                    lamb[:,j] = 0.0
                lamb[:,j] += jac_vecs[f'indeps.znm{j}']

        # compute the final jac vec product
        if 'z_end' in d_outputs:
            for var in d_inputs:
                d_inputs[var] += dfdx[var].dot(d_outputs['z_end'])

    def _set_state(self,step,inputs):
        self._setup_step_backplanes(step,inputs)
        self.problem['modal_solver.zn'] = self.z[step,:]
        self.problem['indeps.time'] = step*self.options['dt']

def check_integrator_partials():
    prob = om.Problem()
    prob.model.add_subsystem('modal_integrator',ModalIntegrator(nmodes=2,nsteps=10,dt=1.0,root_name='sphere_body1'))
    prob.setup()
    prob.check_partials(compact_print=True)

if __name__=='__main__':
    check_integrator_partials()
