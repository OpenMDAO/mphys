This directory contains exploratory work looking at different ways of doing time integration in OpenMDAO

timestep_groups: each time step is a OpenMDAO group that connected within the model.
This approach requires all of the states to be allocated and stored in memory

timestep_loop: time steps are solved in a single explicit component which loops over an OpenMDAO problem that represents the time steps
