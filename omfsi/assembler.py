from __future__ import division, print_function


class OmfsiAssembler(object):
    def add_model_components(self,model,connection_srcs):
        pass

    def add_scenario_components(self,model,connectino_srcs):
        pass

    def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):
        pass

    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        pass

class OmfsiSolverAssembler(OmfsiAssembler):
    def get_nnodes(self):
        return 0

    def get_ndof(self):
        return 3
