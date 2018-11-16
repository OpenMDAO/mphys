from openmdao.api import ExplicitComponent
from mpi4py import MPI

class FuntofemLoadTransfer(ExplicitComponent)
    """
    Component to perform load transfers using MELD
    """
    def initialize(self,meld):

        self.options.declare('meld',default = None,type

        self.options['distributed'] = True

    def setup(self):
        # inputs
        self.add_input('x_s0', desc='initial structural node coordinates')
        self.add_input('x_a0', desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  desc='structural node displacements')
        self.add_input('f_a',  desc='aerodynamic force vector')

        # outputs
        self.add_output('f_s', desc='structural force vector')

        # partials
        self.declare_partials('f_s',['x_s0','x_a0','u_s','f_a'])

    def compute(self, inputs, outputs):
        #u_s  = inputs['u_s']

        f_a =  inputs['f_a']
        f_s = outputs['f_s']

        #self.meld.transferDisps(u_s,u_a)
        self.meld.transferLoads(f_a,f_s)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The MELD residual is defined as:
            L = f_s - g(f_a,u_s,x_a0,x_s0)
        So explicit partials below for f_s are negative parials of L
        """
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_s' in d_inputs:
                # df_s/du_s^T * psi = - dL/du_s^T * psi
                self.meld.applydLduSTrans(-d_outputs['f_s'],d_inputs['u_s']

            if 'f_a' in d_inputs:
                # df_s/df_a^T psi = - dL/df_a^T * psi = -dD/du_s * psi
                prod = np.zeros(d_inputs['f_a'].size)
                self.meld.applydDduS(-d_outputs['f_s'],prod)
                d_inputs['f_a'] += prod

            # df_s/dx_a0^T * psi = - psi^T * dL/dx_s0 in F2F terminology
            if 'x_a0' in d_inputs:
                prod = np.zeros(d_inputs['x_a0'].size)
                self.meld.applydLdxA0(-d_outputs['f_s'],prod)
                d_inputs['x_a0'] += prod

            if 'x_s0' in d_inputs:
                prod = np.zeros(d_inputs['x_s0'].size)
                self.meld.applydLdxS0(-d_outputs['f_s'],d_outputs['x_s0'])
                d_inputs['x_s0'] += prod
