#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np

from openmdao.api import Group, ExplicitComponent

class FsiAssembler(object):
    def __init__(self,struct_assembler,aero_assembler,xfer_assembler,geodisp_assembler=None):
        self.struct_assembler = struct_assembler
        self.aero_assembler   = aero_assembler
        self.xfer_assembler   = xfer_assembler
        if geodisp_assembler is None:
            self.geodisp_assembler = GeoDispAssembler(aero_assembler.solver_dict['nnodes'])
        self.connection_srcs = {}

    def add_model_components(self,model):
        self.geodisp_assembler.add_model_components(model,self.connection_srcs)
        self.xfer_assembler.add_model_components(model,self.connection_srcs)
        self.struct_assembler.add_model_components(model,self.connection_srcs)
        self.aero_assembler.add_model_components(model,self.connection_srcs)

    def add_fsi_subsystem(self,model,scenario):
        fsi_group = scenario.add_subsystem('fsi_group',Group())

        self.geodisp_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)
        self.xfer_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)
        self.struct_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)
        self.aero_assembler.add_fsi_components(model,scenario,fsi_group,self.connection_srcs)

        self.geodisp_assembler.add_scenario_components(model,scenario,self.connection_srcs)
        self.xfer_assembler.add_scenario_components(model,scenario,self.connection_srcs)
        self.struct_assembler.add_scenario_components(model,scenario,self.connection_srcs)
        self.aero_assembler.add_scenario_components(model,scenario,self.connection_srcs)

        self.geodisp_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)
        self.xfer_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)
        self.struct_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)
        self.aero_assembler.connect_inputs(model,scenario,fsi_group,self.connection_srcs)

        return fsi_group


class GeoDispAssembler(object):
    def __init__(self,aero_nnodes):
        self.aero_nnodes = aero_nnodes

    def add_model_components(self,model,connection_srcs):
        pass
    def add_scenario_components(self,model,scenario,connection_srcs):
        pass
    def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):
        fsi_group.add_subsystem('geo_disp',GeoDisp(aero_nnodes=self.aero_nnodes))
        connection_srcs['x_a'] = scenario.name+'.'+fsi_group.name+'.geo_disp.x_a'
    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        model.connect(connection_srcs['x_a0'],scenario.name+'.'+fsi_group.name+'.geo_disp.x_a0')
        model.connect(connection_srcs['u_a'],scenario.name+'.'+fsi_group.name+'.geo_disp.u_a')

class GeoDisp(ExplicitComponent):
    """
    This components is a component that adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options.declare('aero_nnodes',default=None,desc='number of aerodynamic nodes')
        self.options['distributed'] = True

    def setup(self):
        local_size = self.options['aero_nnodes'] * 3
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a0',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')
        self.add_input('u_a', shape=local_size,val=np.zeros(local_size),src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface displacements')

        self.add_output('x_a',shape=local_size,desc='deformed aerodynamic surface')

    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_outputs['x_a'] += d_inputs['x_a0']
                if 'u_a' in d_inputs:
                    d_outputs['x_a'] += d_inputs['u_a']
        if mode == 'rev':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_inputs['x_a0'] += d_outputs['x_a']
                if 'u_a' in d_inputs:
                    d_inputs['u_a']  += d_outputs['x_a']
