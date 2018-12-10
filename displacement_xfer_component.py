from mpi4py import MPI
from openmdao.api import ExplicitComponent
from FUNtoFEM import TransferScheme

class FuntofemDisplacementTransfer(ExplicitComponent)
    """
    Component to perform displacement transfer using MELD
    """
    def initialize(self):
        self.options.declare('disp_xfer_setup', desc='function to instantiate MELD')

        self.options['distributed'] = True

        self.meld = None

    def setup(self):
        # inputs
        self.add_input('x_s0', desc='initial structural node coordinates')
        self.add_input('x_a0', desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  desc='structural node displacements')

        # outputs
        self.add_output('u_a', desc='aerodynamic surface displacements')

        # partials
        self.declare_partials('u_a',['x_s0','x_a0','u_s'])

        # get the transfer scheme object
        self.meld = self.disp_xfer_setup(self.comm)

    def compute(self, inputs, outputs):
        u_s =  inputs['u_s']
        u_a = outputs['u_a']

        if self.meld is None:

            # give MELD the meshes and then form the connections
            x_s0 = inputs['x_s0']
            self.meld.setStructNodes(x_s0)

            x_a0 = inputs['x_a0']
            self.meld.setAeroNodes(x_a0)

            self.meld.initialize()

        self.meld.transferDisps(u_s,u_a)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The MELD residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative parials of D
        """
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'u_s' in d_inputs:
                # du_a/du_s^T * psi = - dD/du_s^T psi
                prod = np.zeros(d_inputs['u_s'].size)
                self.meld.applydDduSTrans(-d_outputs['u_a'],prod)
                d_inputs['u_s'] += prod

            # du_a/dx_a0^T * psi = - psi^T * dD/dx_s0 in F2F terminology
            if 'x_a0' in d_inputs:
                prod = np.zeros(d_inputs['x_a0'].size)
                self.meld.applydDdxA0(-d_outputs['u_a'],prod)
                d_inputs['x_a0'] += prod

            if 'x_s0' in d_inputs:
                self.meld.applydDdxS0(-d_outputs['u_a'],d_inputs['x_s0'])
