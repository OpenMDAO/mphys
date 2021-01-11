from __future__ import division, print_function


class SolverBuilder(object):
    """
    MPHYS builder base class. Template for developers to create their builders.
    """

    def __init__(self, options):
        """
        Initialization routine for the solver builder. This is called by the user in the
        main runscript. The input parameters presented here are not strictly required by
        the mphys api, but they are a good starting point. This is because the user
        initializes this object, and therefore, they have complete freedom on the input
        args.

        Parameters
        ----------
        options : dictionary
            A dictionary containing all the options the user wants to pass to the
            solver object.
        """
        pass

    def init_solver(self, comm):
        """
        The method that initializes the python objects required for the solver.

        Parameters
        ----------
        comm : mpi communicator
            The communicator object created for this xfer object instance.

        """
        pass

    def get_solver(self):
        """
        The method that returns the python objects that represents the solver.

        Returns
        _______
        solver : object
            The solver used inside of the openmdao component that does the calculations
        """
        if hasattr(self, "solver"):
            return self.solver
        else:
            raise NotImplementedError

    def get_element(self, **kwargs):
        """
        Method that returns the openmdao element for this solver

        Parameters
        __________
        **kwargs : optional parameter dictionary
            The builder may or may not propagate the keyword arguments
            provided in this dictionary to the element. These options
            can contain args such as "as_coupling", which is a bool flag
            to determine if the "hooks" required for aerostructural analysis
            should be enabled in the solver. Subject to change

        Returns
        _______
        solver_element : openmdao component or group
            The openmdao element that handles all the computations for
            this solver. This group needs to comply with the MPHYS API
            with the i/o it provides to couple additional physics.
        """
        if hasattr(self, "element"):
            return self.element
        else:
            raise NotImplementedError

    def get_mesh_element(self):
        """
        Method that returns the mesh element for this solver. This method
        is subject to change; however, currently the mesh element it returns
        is connected to the every flight condition that is created using this builder

        Returns
        _______
        mesh_element : openmdao component (or group)
            The openmdao element that stores the original coordinates of the mesh used
            for this solver. For an aerodynamic or structural solver examples, these
            coordinates are the undeflected shape, and they can be used in the context
            of a geometry manipulation routine.
        """
        if hasattr(self, "mesh_element"):
            return self.mesh_element
        else:
            raise NotImplementedError

    def get_mesh_connections(self):
        """
        Method that returns a dictionary of connections between the mesh element and the element
        Returns
        _______
        mesh_element_conns : dict
            The keys of the dictionary represent the components of the element the mesh is connected to and the keys
            represent the variables to connect from the mesh to the element component
        """
        if hasattr(self, "mesh_connections"):
            return self.mesh_connections
        else:
            raise NotImplementedError

    def get_scenario_element(self):
        """
        Method that returns the openmdao element to be added to each scenario

        Returns
        _______
        scenario_element : openmdao component or group
            The openmdao element that handles all the added computations for
            that should be run in each scenario. These may represent functional that
            are run after a coupled analysis is converged.
        """
        if hasattr(self, "scenario_element"):
            return self.scenario_element
        else:
            raise NotImplementedError

    def get_scenario_connections(self):
        """
        Method that returns a dictionary of connections between the scenario element and the element

        Returns
        _______
        scenario_connections : dict
            The keys of the dictionary represent the components of the element the scenario is connected to and the keys
            represent the variables to connect from the scenario to the element component
        """
        if hasattr(self, "scenario_connections"):
            return self.scenario_connections
        else:
            raise NotImplementedError

    def get_nnodes(self):
        """
        Method that returns the number of nodes used in the calculation

        Returns
        _______
        nnodes : int
            number of nodes in the computational domain
        """
        if hasattr(self, "nnodes"):
            return self.nnodes
        else:
            raise NotImplementedError

    def get_ndof(self):
        """
        Method that returns the number of degrees of freedom used at each node.

        Returns
        _______
        ndof : int
            number of degrees of freedom of each node in the computational domain
        """
        if hasattr(self, "ndof"):
            return self.ndof
        else:
            raise NotImplementedError


class DummyBuilder(SolverBuilder):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class XferBuilder(object):
    def __init__(self, options, aero_builder, struct_builder):
        """
        Initialization routine for the xfer builder. This is called by the user in the
        main runscript. This method simply saves the options and the two other solver
        builders that this xfer object will connect. The input parameters we present
        here are not strictly required by the mphys api, but they are a good starting
        point. This is because the user initializes this object, and therefore, they
        have complete freedom on the input args.

        Parameters
        ----------
        options : dictionary
            A dictionary containing all the options the user wants to pass to the
            xfer object.
        aero_builder : solver_builder
            This is the builder for the aero solver this scheme couples
        struct_builder : solver_builder
            This is the builder for the struct solver this scheme couples
        """
        pass

    def init_xfer_object(self, comm):
        """
        The method that initializes the python objects required for the transfer scheme.
        In this method, the xfer builder can make calls to the aero and struct builders
        that are not in the mphys API. E.g. the developers can implement custom methods
        in the respective solver builders to set up the transfer scheme that couples the
        solvers.

        Parameters
        ----------
        comm : mpi communicator
            The communicator object created for this xfer object instance.

        """
        pass

    def get_element(self):
        """
        Method that returns the openmdao elements for the xfer scheme.
        Unlike the get_element methods for a solver_builder, this method returns
        two elements, e.g. one for transfering the displacements and one for
        transfering the loads.

        Returns
        -------
        disp_xfer : Openmdao component (or group)
            The openmdao "element" that is responsible from propagating the displacements
            from the struct solver to the aero solver
        load_xfer : Openmdao component (or group)
            The openmdao "element" that is  responsible from propagating the loads
            from the aero solver to the struct solver
        """
        pass
