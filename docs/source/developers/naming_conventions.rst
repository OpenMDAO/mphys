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
| :code:`T_aero`       | Aerodynamic       |  Aerodynamic surface temperature                                              |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q_aero`       | Aerodynamic       |  Aerodynamic heat flow                                                        |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`x_struct`     | Structural        |  Structural coordinates                                                       |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_struct`     | Structural        |  Structural displacements                                                     |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_struct`     | Structural        |  Structural forces                                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_therm`      | Thermal           |  Thermal (structural) temperature                                             |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q_therm`      | Thermal           |  Thermal (structural) heat flow                                               |
+----------------------+-------------------+-------------------------------------------------------------------------------+

To make swapping solvers easier, it is also helpful to share noncoupling variable names if possible:

+----------------------+-------------------+-------------------------------------------------------------------------------+
| Variable             | Associated Solver | Variable description                                                          |
+======================+===================+===============================================================================+
| :code:`aoa`          | Aerodynamic       |  Aerodynamic angle of attack                                                  |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`yaw`          | Aerodynamic       |  Aerodynamic yaw angle of attack                                              |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`mach`         | Aerodynamic       |  Aerodynamic reference Mach number                                            |
+----------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`reynolds`     | Aerodynamic       |  Aerodynamic reference Reynolds number                                        |
+----------------------+-------------------+-------------------------------------------------------------------------------+

===============
Reference Frame
===============
All vectors are in the global reference frame.
