.. _builders:

########
Builders
########

In large multiphysics problems, creation and connection of the OpenMDAO can be complicated and time-consuming.
The design of Mphys is based on builder classes in order to reduce the burden on the user.
Most of the assembly of the OpenMDAO model with Mphys is handled by a set of builder helper objects.

***************
Solver Builders
***************

The code/discipline assembler are designed for either solvers or transfer scheme modules.
The three tasks common to all the code assemblers are:

.. automodule:: mphys.builder

.. autoclass:: Builder
  :members:
