This directory contains exploratory work looking at different ways of doing time integration in OpenMDAO

prototype/timestep_groups: each time step is a OpenMDAO group that connected within the model.
This approach requires all of the states to be allocated and stored in memory

prototype/timestep_loop: time steps are solved in a single explicit component which loops over an OpenMDAO problem that represents the time steps

builder_version: generalize the timestep_loop to the builder design pattern that we use for steady analysis. Currently the "load and displacement transfer components" are the modal force and displacement conversions. Eventually this should be replaced with an actual LD xfer scheme like MELD and the modal conversions moved inside of the structural subsystem
