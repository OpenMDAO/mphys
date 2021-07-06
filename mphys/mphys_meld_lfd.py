#!/usr/bin/env python

import numpy as np
import openmdao.api as om
from .mphys_meld import MeldBuilder

from funtofem import TransferScheme

class ModeTransfer(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes')
        self.options.declare('nnodes_struct')
        self.options.declare('ndof_struct')
        self.options.declare('nnodes_aero')
        self.options.declare('meld')

    def setup(self):

        self.add_input('x_struct0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])
        self.add_input('mode_shapes_struct', shape_by_conn=True, distributed=True, tags='mphys_coupling')
        self.add_input('x_aero0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])

        nmodes = self.options['nmodes']
        self.nnodes_aero = self.options['nnodes_aero']
        self.nnodes_struct = self.options['nnodes_struct']
        self.ndof_struct = self.options['ndof_struct']

        aero_mode_size = (self.nnodes_aero*3,nmodes)
        self.add_output('mode_shapes_aero', shape=aero_mode_size, distributed=True, tags=['mphys_coupling'])


    def compute(self,inputs,outputs):
        meld = self.options['meld']
        nmodes = self.options['nmodes']
        aero_X = inputs['x_aero0']
        struct_X = inputs['x_struct0']

        aero_modes = np.zeros((aero_X.size,nmodes))
        struct_modes = inputs['mode_shapes_struct'].reshape((-1,nmodes))

        meld.setAeroNodes(aero_X)
        meld.setStructNodes(struct_X)
        meld.initialize()

        struct_mode = np.zeros(self.nnodes_struct*3)
        for mode in range(nmodes):
            struct_mode[0::3] = struct_modes[0::self.ndof_struct,mode]
            struct_mode[1::3] = struct_modes[1::self.ndof_struct,mode]
            struct_mode[2::3] = struct_modes[2::self.ndof_struct,mode]
            aero_mode = np.zeros_like(aero_X)
            meld.transferDisps(struct_mode,aero_mode)
            aero_modes[:,mode] = aero_mode

        outputs['aero_mode_shapes'] = aero_modes.copy()


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        meld = self.options['meld']
        nmodes = self.options['nmodes']
        x_s0 = np.array(inputs['x_struct0'],dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs['x_aero0'],dtype=TransferScheme.dtype)

        meld.setStructNodes(x_s0)
        meld.setAeroNodes(x_a0)

        u_s  = np.zeros(self.nnodes_struct*3,dtype=TransferScheme.dtype)
        for i in range(3):
            u_s[i::3] = inputs['u_struct'][i::self.ndof_struct]
        u_a = np.zeros(self.nnodes_aero*3,dtype=TransferScheme.dtype)

        for mode in range(nmodes):
            u_s  = np.zeros(self.nnodes_struct*3,dtype=TransferScheme.dtype)
            for i in range(3):
                u_s[i::3] = inputs['mode_shapes_struct'][i::self.ndof_struct,mode]
            u_a = np.zeros(self.nnodes_aero*3,dtype=TransferScheme.dtype)
            self.meld.transferDisps(u_s,u_a)
            if mode == 'fwd':
                if 'mode_shapes_aero' in d_outputs:
                    if 'mode_shapes_struct' in d_inputs:
                        d_in = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                        for i in range(3):
                            d_in[i::3] = d_inputs['mode_shapes_struct'][i::self.struct_ndof,mode]
                        prod = np.zeros(self.aero_nnodes*3,dtype=TransferScheme.dtype)
                        self.meld.applydDduS(d_in,prod)
                        d_outputs['mode_shapes_aero'][:,mode] -= np.array(prod,dtype=float)
            if mode == 'rev':
                if 'u_aero' in d_outputs:
                    du_a = np.array(d_outputs['mode_shapes_aero'][:,mode],dtype=TransferScheme.dtype)
                    if 'u_struct' in d_inputs:
                        # du_a/du_s^T * psi = - dD/du_s^T psi
                        prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                        self.meld.applydDduSTrans(du_a,prod)
                        for i in range(3):
                            d_inputs['mode_shapes_struct'][i::self.struct_ndof,mode] -= np.array(prod[i::3],dtype=np.float64)

                    # du_a/dx_a0^T * psi = - psi^T * dD/dx_a0 in F2F terminology
                    if 'x_aero0' in d_inputs:
                        prod = np.zeros(d_inputs['x_aero0'].size,dtype=TransferScheme.dtype)
                        self.meld.applydDdxA0(du_a,prod)
                        d_inputs['x_aero0'] -= np.array(prod,dtype=float)

                    if 'x_struct0' in d_inputs:
                        prod = np.zeros(self.struct_nnodes*3,dtype=TransferScheme.dtype)
                        self.meld.applydDdxS0(du_a,prod)
                        d_inputs['x_struct0'] -= np.array(prod,dtype=float)

class MeldLfdBuilder(MeldBuilder):
    def __init__(self, aero_builder, struct_builder, nmodes,
                       isym=-1, n=200, beta = 0.5, check_partials=False):
            self.nmodes = nmodes
            super().__init__(aero_builder, struct_builder, isym, n, beta, check_partials)

    def get_post_coupling_subsystem(self):
        return ModeTransfer(nmodes = self.nmodes,
                            nnodes_struct = self.nnodes_struct,
                            ndof_struct = self.ndof_struct,
                            nnodes_aero = self.nnodes_aero,
                            meld = self.meld)
