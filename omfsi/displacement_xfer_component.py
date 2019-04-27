import numpy as np

from openmdao.api import ExplicitComponent
from funtofem import TransferScheme

class FuntofemDisplacementTransfer(ExplicitComponent):
    """
    Component to perform displacement transfer using MELD
    """
    def initialize(self):
        self.options.declare('disp_xfer_setup', desc='function to instantiate MELD')

        self.options['distributed'] = True

        self.meld = None
        self.initialized_meld = False

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        # get the transfer scheme object
        disp_xfer_setup = self.options['disp_xfer_setup']
        meld, aero_nnodes, struct_nnodes, struct_ndof = disp_xfer_setup(self.comm)
        self.meld = meld
        self.struct_ndof = struct_ndof
        self.struct_nnodes = struct_nnodes
        self.aero_nnodes = aero_nnodes

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

        # outputs
        self.add_output('u_a', shape = aero_nnodes*3, val=np.zeros(aero_nnodes*3), desc='aerodynamic surface displacements')

        # partials
        #self.declare_partials('u_a',['x_s0','x_a0','u_s'])

    def compute(self, inputs, outputs):
        x_s0 = np.array(inputs['x_s0'],dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs['x_a0'],dtype=TransferScheme.dtype)
        u_a  = np.array(outputs['u_a'],dtype=TransferScheme.dtype)

        u_s  = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs['u_s'][i::self.struct_ndof]

        self.meld.setStructNodes(x_s0)
        self.meld.setAeroNodes(x_a0)

        if not self.initialized_meld:
            self.meld.initialize()
            self.initialized_meld = True

        self.meld.transferDisps(u_s,u_a)

        outputs['u_a'] = u_a

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        The explicit component is defined as:
            u_a = g(u_s,x_a0,x_s0)
        The MELD residual is defined as:
            D = u_a - g(u_s,x_a0,x_s0)
        So explicit partials below for u_a are negative partials of D
        """
        if mode == 'fwd':
            if 'u_a' in d_outputs:
                if 'u_s' in d_inputs:
                    d_in = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    for i in range(3):
                        d_in[i::3] = d_inputs['u_s'][i::self.struct_ndof]
                    prod = np.zeros(self.aero_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDduS(d_in,prod)
                    d_outputs['u_a'] -= np.array(prod,dtype=float)
            else:
                #raise ValueError('forward mode requested but not implemented')
                pass

        if mode == 'rev':
            if 'u_a' in d_outputs:
                du_a = np.array(d_outputs['u_a'],dtype=TransferScheme.dtype)
                if 'u_s' in d_inputs:
                    # du_a/du_s^T * psi = - dD/du_s^T psi
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDduSTrans(du_a,prod)
                    for i in range(3):
                        d_inputs['u_s'][i::self.struct_ndof] -= np.array(prod[i::3],dtype=float)

                # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                if 'x_a0' in d_inputs:
                    prod = np.zeros(d_inputs['x_a0'].size,dtype=TransferScheme.dtype)
                    self.meld.applydDdxA0(du_a,prod)
                    d_inputs['x_a0'] -= np.array(prod,dtype=float)

                if 'x_s0' in d_inputs:
                    prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                    self.meld.applydDdxS0(du_a,prod)
                    d_inputs['x_s0'] -= np.array(prod,dtype=float)
