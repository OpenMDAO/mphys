import numpy as np

from openmdao.api import ExplicitComponent

class FuntofemLoadTransfer(ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):

        self.options.declare('load_xfer_setup',desc='function to set up the load xfer object')

        self.options['distributed'] = True

        self.check_partials = True

    def setup(self):
        self.set_check_partial_options(wrt='*',directional=True)

        # get the transfer scheme object
        load_xfer_setup = self.options['load_xfer_setup']
        meld, aero_nnodes, struct_nnodes, struct_ndof = load_xfer_setup()
        self.meld = meld
        self.struct_ndof   = struct_ndof
        self.struct_nnodes = struct_nnodes
        self.aero_nnodes   =   aero_nnodes

        irank = self.comm.rank

        ax_list = self.comm.allgather(aero_nnodes*3)
        ax1 = np.sum(ax_list[:irank])
        ax2 = np.sum(ax_list[:irank+1])

        sx_list = self.comm.allgather(struct_nnodes*3)
        sx1 = np.sum(sx_list[:irank])
        sx2 = np.sum(sx_list[:irank+1])

        su_list = self.comm.allgather(struct_nnodes*struct_ndof)
        su1 = np.sum(su_list[:irank])
        su2 = np.sum(su_list[:irank+1])

        # inputs
        self.add_input('x_s0', shape = struct_nnodes*3,           src_indices = np.arange(sx1, sx2, dtype=int), desc='initial structural node coordinates')
        self.add_input('x_a0', shape = aero_nnodes*3,             src_indices = np.arange(ax1, ax2, dtype=int), desc='initial aerodynamic surface node coordinates')
        self.add_input('u_s',  shape = struct_nnodes*struct_ndof, src_indices = np.arange(su1, su2, dtype=int), desc='structural node displacements')
        self.add_input('f_a',  shape = aero_nnodes*3,             src_indices = np.arange(ax1, ax2, dtype=int), desc='aerodynamic force vector')

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

        if self.check_partials:
            x_s0 = inputs['x_s0']
            x_a0 = inputs['x_a0']
            self.meld.setStructNodes(x_s0)
            self.meld.setAeroNodes(x_a0)
            #TODO meld needs a set state rather requiring transferDisps to update the internal state
            u_s  = np.zeros(self.struct_nnodes*3)
            for i in range(3):
                u_s[i::3] = inputs['u_s'][i::self.struct_ndof]
            u_a = np.zeros(inputs['f_a'].size)
            self.meld.transferDisps(u_s,u_a)

        self.meld.transferLoads(f_a,f_s)

        outputs['f_s'][:] = 0.0
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
            if 'f_s' in d_outputs:
                if 'u_s' in d_inputs:
                    d_in = np.zeros(self.struct_nnodes*3)
                    for i in range(3):
                        d_in[i::3] = d_inputs['u_s'][i::self.struct_ndof]
                    prod = np.zeros(self.struct_nnodes*3)
                    self.meld.applydLduS(d_in,prod)
                    for i in range(3):
                        d_outputs['f_s'][i::self.struct_ndof] -= prod[i::3]
                if 'f_a' in d_inputs:
                    # df_s/df_a psi = - dL/df_a * psi = -dD/du_s^T * psi
                    prod = np.zeros(self.struct_nnodes*3)
                    self.meld.applydDduSTrans(d_inputs['f_a'],prod)
                    for i in range(3):
                        d_outputs['f_s'][i::self.struct_ndof] -= prod[i::3]
            else:
                #raise ValueError('forward mode requested but not implemented')
                pass

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
