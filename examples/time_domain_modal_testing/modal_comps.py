#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from openmdao.api import ImplicitComponent, ExplicitComponent

class ModalStep(ImplicitComponent):
    """
    Modal structural BDF2 integrator
    Solves one time step of:
      M ddot(z) + C dot(z) + K z - f = 0

    This a simpler model to work on time integration in OMFSI
    It could be an explicit component but it written as an implicit
    since the FEM solvers will be implicit
    """
    def initialize(self):
        self.options.declare('nmodes',default=1)
        self.options.declare('dt',default = 0.0)
    def setup(self):
        # BDF coefficients - 1st and 2nd derivatives
        dt = self.options['dt']
        self.alpha = np.zeros(3)
        self.alpha[0] = 3.0/2.0/dt
        self.alpha[1] = -2.0/dt
        self.alpha[2] = 1.0/2.0/dt

        self.beta = np.zeros(5)
        for i in range(3):
           for j in range(3):
               self.beta[i+j] += self.alpha[i] * self.alpha[j]

        # OM setup
        nmodes = self.options['nmodes']
        self.add_input('m', shape=nmodes, val=np.ones(nmodes), desc = 'modal mass')
        self.add_input('c', shape=nmodes, val=np.ones(nmodes), desc = 'modal damping')
        self.add_input('k', shape=nmodes, val=np.ones(nmodes), desc = 'modal stiffness')

        self.add_input('f', shape=nmodes, val=np.ones(nmodes), desc = 'modal force')

        self.add_input('znm4', shape=nmodes, val=np.ones(nmodes), desc = 'z at step n-4')
        self.add_input('znm3', shape=nmodes, val=np.ones(nmodes), desc = 'z at step n-3')
        self.add_input('znm2', shape=nmodes, val=np.ones(nmodes), desc = 'z at step n-2')
        self.add_input('znm1', shape=nmodes, val=np.zeros(nmodes), desc = 'z at step n-1')

        self.add_output('zn', shape=nmodes, val=np.ones(nmodes), desc = 'current displacement (step n)')

    def _get_accel_and_vel(self,inputs,outputs):
        accel = ( self.beta[0] * outputs['zn']
                + self.beta[1] * inputs['znm1']
                + self.beta[2] * inputs['znm2']
                + self.beta[3] * inputs['znm3']
                + self.beta[4] * inputs['znm4'] )

        vel = ( self.alpha[0] * outputs['zn']
              + self.alpha[1] * inputs['znm1']
              + self.alpha[2] * inputs['znm2'] )
        return accel, vel

    def apply_nonlinear(self, inputs, outputs, residuals):
        accel,vel = self._get_accel_and_vel(inputs,outputs)

        residuals['zn'] = ( inputs['m']*accel
                          + inputs['c']*vel
                          + inputs['k']*outputs['zn']
                          - inputs['f'] )

    def solve_nonlinear(self,inputs,outputs):
        m = inputs['m']
        c = inputs['c']
        k = inputs['k']
        self.m = m
        self.c = c
        self.k = k

        outputs['zn'] = ( inputs['f']
                        - self.beta[1]  * m * inputs['znm1']
                        - self.beta[2]  * m * inputs['znm2']
                        - self.beta[3]  * m * inputs['znm3']
                        - self.beta[4]  * m * inputs['znm4']
                        - self.alpha[1] * c * inputs['znm1']
                        - self.alpha[2] * c * inputs['znm2']
                        ) / (self.beta[0] * m + self.alpha[0] * c + k )

    def solve_linear(self,d_outputs,d_residuals,mode):
        m = self.m
        c = self.c
        k = self.k

        if mode == 'fwd':
            d_outputs['zn'] = d_residuals['zn'] / (self.beta[0] * m + self.alpha[0] * c + k )
        if mode == 'rev':
            d_residuals['zn'] = d_outputs['zn'] / (self.beta[0] * m + self.alpha[0] * c + k )

    def apply_linear(self,inputs,outputs,d_inputs,d_outputs,d_residuals,mode):
        accel, vel = self._get_accel_and_vel(inputs,outputs)
        m = inputs['m']
        c = inputs['c']
        k = inputs['k']

        if mode == 'fwd':
            if 'zn' in d_residuals:
                if 'zn' in d_outputs:
                    d_residuals['zn'] += (self.beta[0] * m + self.alpha[0] * c + k ) * d_outputs['zn']
                if 'm' in d_inputs:
                    d_residuals['zn'] += accel * d_inputs['m']
                if 'c' in d_inputs:
                    d_residuals['zn'] += vel * d_inputs['c']
                if 'k' in d_inputs:
                    d_residuals['zn'] += outputs['zn'] * d_inputs['k']
                if 'f' in d_inputs:
                    d_residuals['zn'] -= d_inputs['f']
                if 'znm4' in d_inputs:
                    d_residuals['zn'] +=  self.beta[4] * m * d_inputs['znm4']
                if 'znm3' in d_inputs:
                    d_residuals['zn'] +=  self.beta[3] * m * d_inputs['znm3']
                if 'znm2' in d_inputs:
                    d_residuals['zn'] +=( self.beta[2] * m * d_inputs['znm2']
                                        + self.alpha[2]* c * d_inputs['znm2'])
                if 'znm1' in d_inputs:
                    d_residuals['zn'] +=( self.beta[1] * m * d_inputs['znm1']
                                        + self.alpha[1]* c * d_inputs['znm1'])
        if mode == 'rev':
            if 'zn' in d_residuals:
                if 'zn' in d_outputs:
                    d_outputs['zn'] += (self.beta[0] * m + self.alpha[0] * c + k ) * d_residuals['zn']
                if 'm' in d_inputs:
                    d_inputs['m'] += accel * d_residuals['zn']
                if 'c' in d_inputs:
                    d_inputs['c'] += vel * d_residuals['zn']
                if 'k' in d_inputs:
                    d_inputs['k'] += outputs['zn'] * d_residuals['zn']
                if 'f' in d_inputs:
                    d_inputs['f'] -= d_residuals['zn']
                if 'znm4' in d_inputs:
                    d_inputs['znm4'] +=  self.beta[4] * m * d_residuals['zn']
                if 'znm3' in d_inputs:
                    d_inputs['znm3'] +=  self.beta[3] * m * d_residuals['zn']
                if 'znm2' in d_inputs:
                    d_inputs['znm2'] +=( self.beta[2] * m * d_residuals['zn']
                                       + self.alpha[2]* c * d_residuals['zn'])
                if 'znm1' in d_inputs:
                    d_inputs['znm1'] +=( self.beta[1] * m * d_residuals['zn']
                                       + self.alpha[1]* c * d_residuals['zn'])

class ModalInterface(ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes',default=1)
        self.options.declare('root_name')
    def _read_mode_shapes(self):
        nmodes = self.options['nmodes']
        for imode in range(nmodes):
            filename = self.options['root_name']+'_mode'+str(imode+1)+'.dat'
            fh = open(filename)
            while True:
                line = fh.readline()
                if 'zone' in line.lower():
                    self.nnodes = int(line.split('=')[2].split(',')[0])
                    if imode == 0:
                        self.mdisp = np.zeros((nmodes,self.nnodes,3))
                    for inode in range(self.nnodes):
                        line = fh.readline()
                        self.mdisp[imode,inode,0] = float(line.split()[4])
                        self.mdisp[imode,inode,1] = float(line.split()[5])
                        self.mdisp[imode,inode,2] = float(line.split()[6])
                if not line:
                    break
            fh.close()

class ModalForces(ModalInterface):
    def setup(self):
        self._read_mode_shapes()
        nmodes = self.options['nmodes']
        nnodes = self.nnodes

        self.add_input('f',shape=nnodes*3,desc = 'nodal force')
        self.add_output('mf',shape=nmodes, desc = 'modal force')

    def compute(self,inputs,outputs):
        outputs['mf'] = 0.0
        for imode in range(self.options['nmodes']):
            for inode in range(self.nnodes):
                for k in range(3):
                    outputs['mf'][imode] += self.mdisp[imode,inode,k] * inputs['f'][3*inode+k]
    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'mf' in d_outputs:
                if 'f' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_outputs['mf'][imode] += self.mdisp[imode,inode,k] * d_inputs['f'][3*inode+k]
        if mode=='rev':
            if 'mf' in d_outputs:
                if 'f' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_inputs['f'][3*inode+k] += self.mdisp[imode,inode,k] * d_outputs['mf'][imode]

class ModalDisplacements(ModalInterface):
    def setup(self):
        self._read_mode_shapes()
        nmodes = self.options['nmodes']
        nnodes = self.nnodes

        self.add_input('md',shape=nmodes, desc = 'modal displacement')
        self.add_output('dx',shape=nnodes*3,desc = 'nodal displacement')

    def compute(self,inputs,outputs):
        outputs['dx'] = 0.0
        for imode in range(self.options['nmodes']):
            for inode in range(self.nnodes):
                for k in range(3):
                    outputs['dx'][3*inode+k] += self.mdisp[imode,inode,k] * inputs['md'][imode]
    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'dx' in d_outputs:
                if 'md' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_outputs['dx'][3*inode+k] += self.mdisp[imode,inode,k] * d_inputs['md'][imode]
        if mode=='rev':
            if 'dx' in d_outputs:
                if 'md' in d_inputs:
                    for imode in range(self.options['nmodes']):
                        for inode in range(self.nnodes):
                            for k in range(3):
                                d_inputs['md'][imode] += self.mdisp[imode,inode,k] * d_outputs['dx'][3*inode+k]

class HarmonicForcer(ExplicitComponent):
    def initialize(self):
        self.options.declare('root_name',default='')
        self.c1 = 1e-3

    def _read_mode_shapes(self):
        filename = self.options['root_name']+'_mode1.dat'
        fh = open(filename)
        while True:
            line = fh.readline()
            if 'zone' in line.lower():
                self.nnodes = int(line.split('=')[2].split(',')[0])
                return

    def setup(self):
        self._read_mode_shapes()

        self.add_input('amp', desc = 'amplitude')
        self.add_input('freq', desc = 'frequency')
        self.add_input('time', desc = 'current time')
        self.add_input('dx',shape=self.nnodes*3)
        self.add_output('f',shape=self.nnodes*3)

    def compute(self,inputs,outputs):
        amp  = inputs['amp']
        freq = inputs['freq']
        time = inputs['time']

        outputs['f'] = amp * np.sin(freq*time) * np.ones(self.nnodes*3) - self.c1 * inputs['dx']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        amp  = inputs['amp']
        freq = inputs['freq']
        time = inputs['time']

        if mode=='fwd':
            if 'f' in d_outputs:
                if 'amp' in d_inputs:
                    d_outputs['f'] += np.sin(freq*time) * np.ones(self.nnodes*3) * d_inputs['amp']
                if 'freq' in d_inputs:
                    d_outputs['f'] += time * np.cos(freq*time) * np.ones(self.nnodes*3) * d_inputs['freq']
                if 'time' in d_inputs:
                    d_outputs['f'] += freq * np.cos(freq*time) * np.ones(self.nnodes*3) * d_inputs['time']
                if 'dx' in d_inputs:
                    d_outputs['f'] -= self.c1 * d_inputs['dx']

        if mode=='rev':
            if 'f' in d_outputs:
                if 'amp' in d_inputs:
                    d_inputs['amp'] += np.sin(freq*time) * np.sum(d_outputs['f'])
                if 'freq' in d_inputs:
                    d_inputs['freq'] += time * np.cos(freq*time) * np.sum(d_outputs['f'])
                if 'time' in d_inputs:
                    d_inputs['time'] += freq * np.cos(freq*time) * np.sum(d_outputs['f'])
                if 'dx' in d_inputs:
                    d_inputs['dx'] -= self.c1 * d_outputs['f']

if __name__ == "__main__":
    from openmdao.api import Problem
    prob = Problem()
    prob.model.add_subsystem('modal_force',ModalForces(nmodes=2,root_name='sphere_body1'))
    prob.model.add_subsystem('modal_step',ModalStep(nmodes=2,dt=0.1))
    prob.model.add_subsystem('modal_disp',ModalDisplacements(nmodes=2,root_name='sphere_body1'))
    prob.model.add_subsystem('harmonic_forcer',HarmonicForcer(root_name='sphere_body1'))

    prob.setup(force_alloc_complex=True)
    prob.check_partials(method='cs',compact_print=True)
