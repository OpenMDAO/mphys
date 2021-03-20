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
        comm : :class:`~mpi4py.MPI.Comm`
            The communicator object created for this xfer object instance.

        """
        pass

    def get_mesh_coordinate_subsystem(self):
        """
        The subsystem that contains the subsystem that will return the mesh
        coordinates

        Returns
        -------
        mesh : :class:`~openmdao.api.Component` or :class:`~openmdao.api.Group`
            The openmdao subsystem that has an output of coordinates.
        """
        return None

    def get_coupling_group_subsystem(self):
        """
        The subsystem that this builder will add to the CouplingGroup

        Returns
        -------
        subsystem : :class:`~openmdao.api.Component` or :class:`~openmdao.api.Group`
            The openmdao subsystem that handles all the computations for
            this solver. Transfer schemes can return multiple subsystems
        """
        return None

    def get_pre_coupling_subsystem(self):
        """
        Method that returns the openmdao subsystem to be added to each scenario before the coupling group

        Returns
        -------
        subsystem : :class:`~openmdao.api.Component` or :class:`~openmdao.api.Group`
        """
        return None

    def get_post_coupling_subsystem(self):
        """
        Method that returns the openmdao subsystem to be added to each scenario after the coupling group

        Returns
        -------
        subsystem : :class:`~openmdao.api.Component` or :class:`~openmdao.api.Group`
        """
        return None

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
