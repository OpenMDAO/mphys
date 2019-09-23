# OMFSI - OpenMDAO FSI

This repository contains OpenMDAO components and related code for fluid-structure interaction problems in OpenMDAO.

# Current Design

The structure of the coupled model has three basic levels: model, scenario, and fsi.
The fsi group is the physics coupling level.
It contains the disciplinary solvers and data transfer components.
The scenario group is a point in a multipoint optimization.
The scenario group would contain the fsi group plus any components that are condition/solution specific that only need to be computed once at the beginning or end of the coupled solve, e.g., function evaluators like lift or max stress calculators.
The model group is the model as defined by the OpenMDAO problem.
It would contain the scenario groups and any other components that deal with data going to or coming from multiple scenarios.
For example a geometry or shape parameterization component.

The construction of the coupled system is handled by assembler objects.
The FSI Assembler is the highest level assembler, and each code provides its own lower level assembler.
All the code level assemblers should perform 3 tasks. 1) Add its components to the appropriate level 2) Tell the other solvers about its outputs by adding them to the `connection_srcs` dictionary 3) Connect its components' inputs from the `connection_srcs` dictionary.
Some assemblers like the structures assembler need to provide additional functionality like `get_nnodes` or `get_ndof` because the transfer scheme components need to know the size of the vectors during setup.

## Variable conventions

Currently all of the variables are flattened arrays.
The variables are named according to our convection in FUNtoFEM.
I'm not strongly tied to that naming convection if there is another preference in the group.

# TACS usage

For usage of the TACS components, it is assumed that the user is reading a BDF file.
The user defines a dictionary that is given to the assembler.
This dictionary contains the BDF file name and pointers to functions that do problem-specific setup like adding the elements, setting up the functions, and setting up the f5 file writing.
See examples/tacs\_only for an setup of a uCRM wingbox analysis.


# FUNtoFEM

The load and displacement transfer schemes are implemented as explicit components. These components need information like the disciplines' comms and number of aero surface and structural nodes on each processor which is pulls from aerodynamic and structures assemblers.
