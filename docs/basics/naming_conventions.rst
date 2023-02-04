***************************
Variable Naming Conventions
***************************

While it is possible to set up the same OpenMDAO multiphysics problem with different sets of variable names, it is preferable for codes solving the same physics to use the same variable names to be more easily interchangeable.
This table provides the required names for coupling variables associated with a particular physics in MPhys.

+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| Variable             | Associated Solver | MPhys tag         | Variable description                                                          |
+======================+===================+===================+===============================================================================+
| :code:`x_aero0`      | Aerodynamic       | mphys_coordinates |  Aerodynamic surface coordinates (jig shape)                                  |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`x_aero`       | Aerodynamic       | mphys_coupling    |  Aerodynamic surface coordinates (deformed)                                   |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_aero`       | Aerodynamic       | mphys_coupling    |  Aerodynamic surface displacements                                            |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_aero`       | Aerodynamic       | mphys_coupling    |  Aerodynamic surface forces                                                   |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_convect`    | Aerodynamic       | mphys_coupling    |  Temperature for convective solver at interface                               |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q_convect`    | Aerodynamic       | mphys_coupling    |  Convective heat flow at interface                                            |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`x_struct0`    | Structural        | mphys_coordinates |  Structural coordinates (jig shape)                                           |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`u_struct`     | Structural        | mphys_coupling    |  Structural state vector (displacements)                                      |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`f_struct`     | Structural        | mphys_coupling    |  Structural forces                                                            |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`T_conduct`    | Thermal           | mphys_coupling    |  Temperature at interface (structural side)                                   |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+
| :code:`q_conduct`    | Thermal           | mphys_coupling    |  Conductive heat flow at interface (structural side)                          |
+----------------------+-------------------+-------------------+-------------------------------------------------------------------------------+

To make swapping solvers easier, it is also helpful to share noncoupling variable names if possible:

+----------------------+-------------------+-------------------+---------------------------------------------------------------------------------+
| Variable             | Associated Solver | MPhys tag         | Variable description                                                            |
+======================+===================+===================+=================================================================================+
| :code:`aoa`          | Aerodynamic       | mphys_input       |  Angle of attack (please include units='deg' or 'rad' when declared)            |
+----------------------+-------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`yaw`          | Aerodynamic       | mphys_input       |  Yaw angle  (please include units='deg' or 'rad' when declared)                 |
+----------------------+-------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`mach`         | Aerodynamic       | mphys_input       |  Reference Mach number                                                          |
+----------------------+-------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`reynolds`     | Aerodynamic       | mphys_input       |  Reference Reynolds number                                                      |
+----------------------+-------------------+-------------------+---------------------------------------------------------------------------------+
| :code:`q_inf`        | Aerodynamic       | mphys_input       |  Dynamic pressure                                                               |
+----------------------+-------------------+-------------------+---------------------------------------------------------------------------------+
