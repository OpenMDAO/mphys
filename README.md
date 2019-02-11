# OMFSI - OpenMDAO FSI

This repository contains OpenMDAO components and related code for fluid-structure interaction problems in OpenMDAO.

# Current Design

The current set up is such that assembler.py contains a class to store data and objects that multiple components need but may not be available during the initialize phase of the OpenMDAO problem.
The OpenMDAO components that need to access information from this assembler are given a member function of this object as an option.
During the 'setup' phase of the OpenMDAO problem, the components can call this function to get the info it needs.

One limitation with the current implementation is that the assembler is specific to the current codes. Swapping FUN3D for ADflow would require a different assembler.


# FSI coupling methodology

Generating the n2 diagram from the funtofem\_and\_tacs example will show the planned coupling connections between the FSI components. The variable naming essentially following Jacobson et al 2018 "An Aeroelastic Coupling Framework for Time-accurate Aeroelastic Analysis and Optimization"


# TACS usage

For usage of the TACS components, it is assumed that the user is reading a BDF file.
The user defines a dictionary that is given to the assembler.
This dictionary contains the BDF file name, the number of TACS processors, a list of the tacs functions to evaluate, and an add\_elements function.
The add\_elements function is given a TACS mesh object, and the user prescribes the element and constitutive objects for each component in the mesh object.
See examples/tacs\_only for an setup of a uCRM wingbox analysis.


# FUNtoFEM

The load and displacement transfer schemes are implemented as explicit components. These components need information like the disciplines' comms and number of aero surface and structural nodes on each processor.
