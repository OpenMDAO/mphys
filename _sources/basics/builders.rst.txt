.. _builders:

########
Builders
########

In large multiphysics problems, creation and connection of the OpenMDAO can be complicated and time-consuming.
The design of Mphys is based on builder classes in order to reduce the burden on the user.
Most of the assembly of the OpenMDAO model with Mphys is handled by a set of builder helper objects.

Developers wishing to integrate their code to Mphys should subclass the builder and implement the methods relevant to their code.
Not all builders need to implement all the methods.
For example, a transfer scheme builder may not need a precoupling post coupling subsystem in the scenario.


.. automodule:: mphys.builder

.. autoclass:: Builder
  :members:
