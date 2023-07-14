class ServerManager:
    """
    A class used by the client-side RemoteComp to facilitate communication
    with the remote, server-side OpenMDAO problem.

    To make a particular derived class, implement the start_server,
    stop_server, and enough_time_is_remaining functions.
    """
    def start_server(self):
        pass

    def stop_server(self):
        pass

    def enough_time_is_remaining(self, estimated_model_time):
        return True
