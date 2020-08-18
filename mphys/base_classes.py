from __future__ import division, print_function

class SolverBuilder(object):
    """
    MPHYS builder base class. Template for developers to create their builders.
    """

    def __init__(self, options):
        '''
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
        '''
        pass

    def init_solver(self, comm):
        '''
        The method that initializes the python objects required for the solver.

        Parameters
        ----------
        comm : mpi communicator
            The communicator object created for this xfer object instance.

        '''
        pass

    def get_element(self, **kwargs):
        '''
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
        '''
        pass

    def get_mesh_element(self):
        '''
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
        '''
        pass


class XferBuilder(object):

    def __init__(self, options, aero_builder, struct_builder):
        '''
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
        '''
        pass

    def init_xfer_object(self, comm):
        '''
        The method that initializes the python objects required for the transfer scheme.
        In this method, the xfer builder can make calls to the aero and struct builders
        that are not in the mphys API. E.g. the developers can implement custom methods
        in the respective solver builders to set up the transfer scheme that couples the
        solvers.

        Parameters
        ----------
        comm : mpi communicator
            The communicator object created for this xfer object instance.

        '''
        pass

    def get_element(self):
        '''
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
        '''
        pass
