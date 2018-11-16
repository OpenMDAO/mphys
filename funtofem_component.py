from openmdao.api import ExplicitComponent
from mpi4py import MPI

class FuntofemTransfer(ExplicitComponent)
    """
    Component to perform both load and displacement transfers using MELD

    Assumptions:
        single body, serial execution

    """
    def initialize(self):
        self.options.declare('beta', default = 0.5, type=float, desc='exponential decay factor')
        self.options.declare('n',    default = 200, type=int,   desc='number of struct. nodes attached to each aero node')
        self.options.declare('isym', default =  -1, type=int,   desc='symmetry plane ')

        self.meld = None

    def setup(self):
        # inputs
        self.add_input('x_s0', 0.0, desc='initial structural node coordinates')
        self.add_input('x_a0', 0.0, desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  0.0, desc='structural node displacements')
        self.add_input('f_a',  0.0, desc='aerodynamic force vector')

        # outputs
        self.add_output('u_a',0.0, desc='aerodynamic surface displacements')
        self.add_output('f_s',0.0, desc='structural force vector')

        # partials
        self.declare_partials('u_a',['x_s0','x_a0','u_s'])

        self.declare_partials('f_s',['x_s0','x_a0','u_s','f_a'])

    def compute(self, inputs, outputs):
        u_s  = inputs['u_s']
        f_a  = inputs['f_a']

        u_a = outputs['u_a']
        f_s = outputs['f_s']

        if self.meld is None:
            # instantiate the transfer scheme
            comm = MPI.COMM_WORLD
            isym = self.options['isym']
            n    = self.options['n']
            beta = self.options['beta']
            self.meld = TransferScheme.pyMELD(comm,comm,0,comm,0,isym,n,beta)

            # give MELD the meshes and then form the connections
            x_s0 = inputs['x_s0']
            self.meld.setStructNodes(x_s0)

            x_a0 = inputs['x_a0']
            self.meld.setAeroNodes(x_a0)

            self.meld.initialize()

        self.meld.transferDisps(u_s,u_a)
        self.meld.transferLoads(f_a,f_s)


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':

        if mode == 'rev':
            # displacement transfer
            # du_a/du_s^T * psi = - dD/du_s^T psi
            self.meld.applydDduSTrans(-d_outputs['u_a'],d_outputs['u_s'])

            # du_a/dx_a0^T * psi = - psi^T * dD/dx_s0 in F2F terminology
            self.meld.applydDdxA0(-d_outputs['u_a'],d_outputs['x_a0'])
            self.meld.applydDdxS0(-d_outputs['u_a'],d_outputs['x_s0'])

            # force transfer
            # df_s/df_a^T * psi = - dL/df_a^T * psi = -dD/du_s * psi
            self.meld.applydDduS(-d_outputs['f_s'],d_inputs['f_a']

            # df_s/du_s^T * psi = - dL/du_s^T * psi
            self.meld.applydLduSTrans(-d_outputs['f_s'],d_inputs['u_s']

            # df_s/dx_a0^T * psi = - psi^T * dL/dx_s0 in F2F terminology
            self.meld.applydLdxA0(-d_outputs['f_s'],d_outputs['x_a0'])
            self.meld.applydLdxS0(-d_outputs['f_s'],d_outputs['x_s0'])
