
class DummyBuilder(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

     # api level method for all builders
    def init_solver(self, comm):
        pass

    # api level method for all builders
    def get_solver(self):
        if hasattr(self, 'solver'):
            return self.solver
        else:
            raise NotImplementedError
    

    # api level method for all builders
    def get_element(self, **kwargs):
        if hasattr(self, 'element'):
            return self.element
        else:
            raise NotImplementedError


    def get_mesh_element(self):
        if hasattr(self, 'mesh_element'):
            return self.mesh_element
        else:
            raise NotImplementedError

    def get_scenario_element(self):
        if hasattr(self, 'scenario_element'):
            return self.scenario_element
        else:
            raise NotImplementedError

    def get_scenario_connections(self):
        if hasattr(self, 'scenario_connections'):
            return self.scenario_connections
        else:
            raise NotImplementedError


    def get_mesh_connections(self):
        if hasattr(self, 'mesh_connections'):
            return self.mesh_connections
        else:
            raise NotImplementedError

    def get_nnodes(self):
        if hasattr(self, 'nnodes'):
            return self.nnodes
        else:
            raise NotImplementedError

    def get_ndof(self):
        if hasattr(self, 'ndof'):
            return self.ndof
        else:
            raise NotImplementedError
