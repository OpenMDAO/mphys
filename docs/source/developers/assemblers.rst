**********
Assemblers
**********

In large multiphysics problems, creation and connection of the OpenMDAO can be complicated and time-consuming.
The design of OMFSI is based on assembler classes in order to reduce the burden on the user.
The most of the assembly of OpenMDAO model with OMFSI is handled by a set of assembler helper objects.
There is a primary :code:`fsi_assembler` as well as one associated with each code/discipline connected with OMFSI.
The code/discipline assembler each must perform three tasks.
Some disciplines have some discipline-specific functionality required as well.
The three tasks common to all the code assemblers are:

#. Define functions to add its components to the appropriate group levels in the model hierarchy.
#. Identify the outputs of its components by adding them the :code:`connection_srcs` dictionary;
#. Connect its components' inputs from the :code:`connection_srcs` dictionary.

The aerodynamic assembler must provide two additional getter functions: :code:`get_comm` and :code:`get_nnodes` which get the mpi communicator and number of nodes owned by the local processor, respectively.

The structural assembler must provide three additional getter functions: :code:`get_comm`, :code:`get_nnodes`, and :code:`get_ndof` which get the mpi communicator, number of nodes owned by the local processor, and the number of degrees of freedom per node, respectively.
