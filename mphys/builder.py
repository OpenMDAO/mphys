class Builder:
    """
    MPHYS builder base class. Template for developers to create their builders.
    """

    def __init__(self):
        """
        Because the MPI communicator that will be used inside the OpenMDAO
        problem is not known when the builder is instantiated. The actual
        solver, transfer scheme, etc. should not be instantiated in the
        constructor.
        """
        self.solver = None

    def initialize(self, comm):
        """
        Initialize the solver, transfer scheme, etc.
        This method will be called when the MPI comm is available

        Parameters
        ----------
        comm : ~mpi4py.MPI.Comm
            The communicator object created for this xfer object instance.

        """
        pass

    def get_mesh_coordinate_subsytem(self):
        return None

    def get_coupling_group_subsystem(self):
        """
        The element that this builder will add to the CouplingGroup

        Returns
        -------
        solver_element : ~openmdao.api.Component or ~openmdao.api.Group
            The openmdao element that handles all the computations for
            this solver. Transfer schemes can return multiple elements
        """
        return None

    def get_scenario_subsystems(self):
        """
        Method that returns the openmdao element to be added to each scenario

        Returns
        -------
        pre_coupling_element : openmdao.api.Component or ~openmdao.api.Group
            The openmdao element that handles all the added computations for
            that should be run in each scenario. These may represent functional that
            are run after a coupled analysis is converged.
        """
        pre_coupling_element = None
        post_coupling_element = None
        return pre_coupling_element, post_coupling_element

    def get_number_of_nodes(self):
        """
        Method that returns the number of nodes defining the interface
        (input) mesh


        Returns
        -------
        number_of_nodes : int
            number of nodes in the computational domain
        """
        return None

    def get_ndof(self):
        """
        The number of degrees of freedom used at each output location.

        Returns
        -------
        ndof : int
            number of degrees of freedom of each node in the domain
        """
        return None
