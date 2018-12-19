import numpy as np

from openmdao.api import ExplicitComponent

class FuntofemLoadTransfer(ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):

        self.options.declare('load_xfer_setup',desc='function to set up the load xfer object')

        self.options['distributed'] = True

    def setup(self):
        # get the transfer scheme object
        load_xfer_setup = self.options['load_xfer_setup']
        meld, aero_nnodes, struct_nnodes, struct_ndof = load_xfer_setup()
        self.meld = meld
        self.struct_ndof   = struct_ndof
        self.struct_nnodes = struct_nnodes
        self.aero_nnodes   =   aero_nnodes

        # inputs
        self.add_input('x_s0', shape = struct_nnodes*3,           desc='initial structural node coordinates')
        self.add_input('x_a0', shape = aero_nnodes*3,             desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  shape = struct_nnodes*struct_ndof, desc='structural node displacements')
        self.add_input('f_a',  shape = aero_nnodes*3,             desc='aerodynamic force vector')

        # outputs
        self.add_output('f_s', shape = struct_nnodes*struct_ndof, desc='structural force vector')

        # partials
        #self.declare_partials('f_s',['x_s0','x_a0','u_s','f_a'])

    def compute(self, inputs, outputs):
        u_s  = np.zeros(self.struct_nnodes*3)
        for i in range(3):
            u_s[i::3] = inputs['u_s'][i::self.struct_ndof]


        f_a =  inputs['f_a']
        f_s = np.zeros(self.struct_nnodes*3)

        #TODO meld needs a set state rather requiring transferDisps to update the internal state
        u_a  = np.zeros(inputs['f_a'].size)
        self.meld.transferDisps(u_s,u_a)
        self.meld.transferLoads(f_a,f_s)

        for i in range(3):
            outputs['f_s'][i::self.struct_ndof] = f_s[i::3]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            f_s = g(f_a,u_s,x_a0,x_s0)
        The MELD internal residual is defined as:
            L = f_s - g(f_a,u_s,x_a0,x_s0)
        So explicit partials below for f_s are negative partials of L
        """
        if mode == 'fwd':
            raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            if 'f_s' in d_outputs:
                d_out = np.zeros(self.struct_nnodes*3)
                for i in range(3):
                    d_out[i::3] = d_outputs['f_s'][i::self.struct_ndof]

                if 'u_s' in d_inputs:
                    d_in = np.zeros(self.struct_nnodes*3)
                    # df_s/du_s^T * psi = - dL/du_s^T * psi
                    self.meld.applydLduSTrans(d_out,d_in)

                    for i in range(3):
                        d_inputs['u_s'][i::self.struct_ndof] -= d_in[i::3]

                if 'f_a' in d_inputs:
                    # df_s/df_a^T psi = - dL/df_a^T * psi = -dD/du_s * psi
                    prod = np.zeros(self.aero_nnodes*3)
                    self.meld.applydDduS(d_out,prod)
                    d_inputs['f_a'] -= prod

                if 'x_a0' in d_inputs:
                    # df_s/dx_a0^T * psi = - psi^T * dL/dx_a0 in F2F terminology
                    prod = np.zeros(self.aero_nnodes*3)
                    self.meld.applydLdxA0(d_out,prod)
                    d_inputs['x_a0'] -= prod

                if 'x_s0' in d_inputs:
                    # df_s/dx_s0^T * psi = - psi^T * dL/dx_s0 in F2F terminology
                    prod = np.zeros(self.struct_nnodes*3)
                    self.meld.applydLdxS0(d_out,prod)
                    d_inputs['x_s0'] -= prod
