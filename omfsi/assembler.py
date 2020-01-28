from __future__ import division, print_function


class OmfsiAssembler(object):
    """
    OMFSI Assembler base class. Template for developers to add their code to an
    OMFSI problem.
    """
    def add_model_components(self,model,connection_srcs):
        """
        Add solver modules to the highest level of the openmdao model.
        The function should add components or groups to the model and identify
        any outputs of these modules in the connection_srcs dictionary.

        Parameters
        ----------
        model : openmdao model
            The OpenMDAO model associated with the design problem

        connection_srcs : dict
            A dictionary with keys of variables and the corresponding entries
            being the model path for their source, i.e., component that outputs
            this variable.
        """
        pass

    def add_scenario_components(self,model,scenario,connection_srcs):
        """
        Add solver modules to the scenario level of the openmdao model.
        The function should add components or groups to the model and identify
        any outputs of these modules in the connection_srcs dictionary.

        Parameters
        ----------
        model : openmdao model
            The OpenMDAO model associated with the design problem

        scenario : openmdao group
            The OpenMDAO group associated with the design scenario

        connection_srcs : dict
            A dictionary with keys of variables and the corresponding entries
            being the model path for their source, i.e., component that outputs
            this variable.
        """
        pass

    def add_fsi_components(self,model,scenario,fsi_group,connection_srcs):
        """
        Add solver modules to the solver coupling level of the openmdao model.
        The function should add components or groups to the model and identify
        any outputs of these modules in the connection_srcs dictionary.

        Parameters
        ----------
        model : openmdao model
            The OpenMDAO model associated with the design problem

        scenario : openmdao group
            The OpenMDAO group associated with the design scenario

        fsi_group : openmdao group
            The OpenMDAO group associated with the solver coupling

        connection_srcs : dict
            A dictionary with keys of variables and the corresponding entries
            being the model path for their source, i.e., component that outputs
            this variable.
        """
        pass

    def connect_inputs(self,model,scenario,fsi_group,connection_srcs):
        """
        Use the connection_srcs dictionary to get paths for the inputs of the
        components that this assembler has added to the OpenMDAO problem.

        Parameters
        ----------
        model : openmdao model
            The OpenMDAO model associated with the design problem

        scenario : openmdao group
            The OpenMDAO group associated with the design scenario

        fsi_group : openmdao group
            The OpenMDAO group associated with the solver coupling

        connection_srcs : dict
            A dictionary with keys of variables and the corresponding entries
            being the model path for their source, i.e., component that outputs
            this variable.
        """
        pass

class OmfsiSolverAssembler(OmfsiAssembler):
    """
    OMFSI Assembler base class for solvers. In addition to the standard OMFSI
    Assembler base class member functions, solvers should implement the following
    functions to allow interactions with transfer scheme modules.
    """
    def get_nnodes(self):
        """
        Getter to retrieve the number of nodes owned by the local processor.

        Returns
        -------
        nnodes : int
            The local number of nodes
        """
        return 0

    def get_ndof(self):
        """
        Getter to retrieve the number of degrees of freedom of this solver's
        output state. Note, this is the number of degrees of the output associated
        with the coupling, so a CFD code should return the number of degrees of 
        freedom of the surface loads, not the flow state vector size if FSI coupling is
        being performed.

        Returns
        -------
        ndof : int
            The number of degrees of freedom per node
        """
        return 3

    def get_comm(self):
        """
        Getter to retrieve the communicator used by the solver.

        Returns
        -------
        comm : MPI communicator
            The solver's mpi communicator  
        """
        return 
