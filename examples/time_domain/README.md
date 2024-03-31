# Time-domain Prototyping

Note: these demonstrations are initial proofs of concept and have not been exercised with actual high-fidelity codes. Time-domain infrastructure and standards in MPhys may need to be changed as issues appear when actual codes are integrated.

This directory contains exploratory work looking at two different ways of doing time integration in OpenMDAO.


1. `timestep_groups`: The first mode is the default mode of OpenMDAO where each time step of the temporal problem is a subsystem and connections are used to indicate the dependency of a particular time step on previous time
This requires OpenMDAO to allocate and store all states in the system in memory at all times;
this is not practical with large high-fidelity codes where there could be 100s or 1000s of time steps.

2. `timestep_loop`: The alternative approach is to use nested OpenMDAO problems.
The outer OpenMDAO problem is the optimization problem, and the inner problem represents a single coupled time step of the time-domain problem.
There is an explicit component in the outer problem that calls the inner problem in a time step loop in order to solve the time-domain problem.
This explicit component does not expose the temporally varying states to the outer problem, and it manages the temporal data in such a way that it doesn't need to keep all of it in memory at the same time, i.e., it can write and read states to/from disk as needed in order to solve the temporal nonlinear, forward, or reverse problems.

These two modes are demonstrated in the `prototype` directory which models a simplified structural dynamics problem with a weakly coupled forcing component.
The purpose of these prototypes is to demonstrate that the `timestep_loop` method will get the same answer as the default `timestep_loop` approach while not having to expose every temporal state to the outer OpenMDAO problem.

While the prototype shows that the `timestep_loop` approach is feasible, the temporal integrator component is specific to the structural problem being solved.
Therefore, the `builder_version` looks to generalize this timestep loop approach by introducing unsteady versions of the MPhys Builders and Scenarios called the Integrator.
The TimeDomainBuilder extends the standard MPhys Builder class to provide additional information about the temporal integration such as how many temporal backplanes of data are required for a particular state.
The Integrator component performs the time step loops and manages the temporal state data including shuffling backplanes of states as the time is advanced.
In this simplified example, the aerodynamic loads are computed on the same coordinate locations as the structural mesh.
Since a normal load and displacement transfer scheme is not required,
the example's "load and displacement transfer components" are the modal force and displacement conversions in order to fit a aerostructural coupling group pattern.
