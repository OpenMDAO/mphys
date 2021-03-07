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

#. Define functions to add its components to the appropriate group levels in the model hierarchy.
#. Identify the outputs of its components by adding them the :code:`connection_srcs` dictionary;
#. Connect its components' inputs from the :code:`connection_srcs` dictionary.

The solver assemblers must provide three additional getter functions: :code:`get_comm`, :code:`get_nnodes`, and :code:`get_ndof` which get the mpi communicator, number of nodes owned by the local processor, and the number of degrees of freedom per node, respectively.
Thse getter functions allow transfer scheme components


.. automodule:: omfsi.assembler

.. autoclass:: OmfsiAssembler
  :members:

.. autoclass:: OmfsiSolverAssembler
  :members:
  :show-inheritance:

**************************
Coupling Problem Assembler
**************************
The coupling problem assemblers wrap the code assemblers into smaller functions that the user can use to build their model.

.. automodule:: omfsi.fsi_assembler


.. autoclass:: FsiAssembler
  :members:
