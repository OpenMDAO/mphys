******************
OMFSI Conventions
******************


While it is possible to set up the same OpenMDAO multiphysics problem with different sets of variable names, it is preferable for codes solving the same physics to use the same variable names to be more easily interchangeable.

====================
Variable Conventions
====================
The current variable naming convention is based on `FUNtoFEM <https://arc.aiaa.org/doi/10.2514/6.2018-0100>`_, but with some modifications.

- All vectors are flattened in row-major ordering, e.g., a set of coordinates would be :math:`x_0,y_0,z_0,x_1,y_1,z_1,...`.
- All vectors are in the global reference.

This table provides the required for coupling a particular physics OMFSI.

+----------------------+-------------------+-------------------------------------------------------------------------------+
| Variable             | Associated Solver | Variable description                                                          |
+======================+===================+===============================================================================+
| :code:`x_a`          | Aerodynamic       |  Aerodynamic surface coordinates                                              |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_a`          | Aerodynamic       |  Aerodynamic surface displacements                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_a`          | Aerodynamic       |  Aerodynamic surface forces                                                   |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_a`          | Aerodynamic       |  Aerodynamic surface temperature                                              |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`phi_a`        | Aerodynamic       |  Aerodynamic heat flow                                                        |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`x_s`          | Structural        |  Structural coordinates                                                       |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_s`          | Structural        |  Structural displacements                                                     |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_s`          | Structural        |  Structural forces                                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_s`          | Structural        |  Structural temperature                                                       |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`phi_s`        | Structural        |  Structural heat flow                                                         |
+----------------------+-------------------+-------------------------------------------------------------------------------+


Other variables may only exist in higher fidelity solvers, but may be required for coupling to certain physics.
For example, the flow state in the volume domain is not utilized for FSI coupling, but may be required for coupling to an aeroacoustic tool.
The following table contains some of these types of variables.

+----------------------+-------------------+-------------------------------------------------------------------------------+
| Variable             | Associated Solver | Variable description                                                          |
+======================+===================+===============================================================================+
| :code:`x_g`          | Aerodynamic       |  Aerodynamic volume mesh coordinates                                          |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q`            | Aerodynamic       |  Aerodynamic state vector                                                     |
+----------------------+-------------------+-------------------------------------------------------------------------------+

===================
Solver Block Naming
===================

Following the variable naming is critical for OMFSI to work, but standardizing the names of the solver and transfer scheme blocks in the FSI coupling level is helpful too.
This allows the OMFSI assemblers to reorder solvers in order to make the coupled nonlinear and linear solvers more efficient.

===============
Reference Frame
===============
All vectors are in the global reference frame.
