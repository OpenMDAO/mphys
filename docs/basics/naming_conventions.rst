******************
Mphys Conventions
******************

While it is possible to set up the same OpenMDAO multiphysics problem with different sets of variable names, it is preferable for codes solving the same physics to use the same variable names to be more easily interchangeable.

====================
Variable Conventions
====================
The current variable naming convention is based on `FUNtoFEM <https://arc.aiaa.org/doi/10.2514/6.2018-0100>`_, but with some modifications.

- All vectors are in the global reference.

This table provides the required names for coupling variables associated with a particular physics in Mphys.

+----------------------+-------------------+-------------------------------------------------------------------------------+
| Variable             | Associated Solver | Variable description                                                          |
+======================+===================+===============================================================================+
| :code:`x_aero`       | Aerodynamic       |  Aerodynamic surface coordinates                                              |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_aero`       | Aerodynamic       |  Aerodynamic surface displacements                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_aero`       | Aerodynamic       |  Aerodynamic surface forces                                                   |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_convect`    | Aerodynamic       |  Temperature for convective solver at interface                               |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q_convect`    | Aerodynamic       |  Convective heat flow at interface                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`x_struct`     | Structural        |  Structural coordinates                                                       |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_struct`     | Structural        |  Structural state vector (displacements)                                      |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_struct`     | Structural        |  Structural forces                                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_conduct`    | Thermal           |  Temperature at interface (structural side)                                   |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q_conduct`    | Thermal           |  Conductive heat flow at interface (structural side)                          |
+----------------------+-------------------+-------------------------------------------------------------------------------+

To make swapping solvers easier, it is also helpful to share noncoupling variable names if possible:

+----------------------+-------------------+---------------------------------------------------------------------------------+
| Variable             | Associated Solver | Variable description                                                            |
+======================+===================+=================================================================================+
| :code:`aoa`          | Aerodynamic       |  Aerodynamic angle of attack (please include units='deg' or 'rad' when declared)|
+----------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`yaw`          | Aerodynamic       |  Aerodynamic yaw angle  (please include units='deg' or 'rad' when declared)     |
+----------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`mach`         | Aerodynamic       |  Aerodynamic reference Mach number                                              |
+----------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`reynolds`     | Aerodynamic       |  Aerodynamic reference Reynolds number                                          |
+----------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`q_inf`        | Aerodynamic       |  Aerodynamic dynamic pressure                                                   |
+----------------------+-------------------+---------------------------------------------------------------------------------+

===============
Reference Frame
===============
All vectors are in the global reference frame.
