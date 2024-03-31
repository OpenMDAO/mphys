class ServerManager:
    """
    A class used by the client-side RemoteComp to facilitate communication
    with the remote, server-side OpenMDAO problem.

    To make a particular derived class, implement the start_server,
    stop_server, and enough_time_is_remaining functions.
    """
    def start_server(self):
        """
        Start the server.
        """
        pass

    def stop_server(self):
        """
        Stop the server.
        """
        pass

    def enough_time_is_remaining(self, estimated_model_time):
        """
        Check if the current HPC job has enough time remaining
        to run the next analysis.

        Parameters
        ----------
        estimated_model_time : float
            How much time the new analysis is estimated to take
        """
        return True
