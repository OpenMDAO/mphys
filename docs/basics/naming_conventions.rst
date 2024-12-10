===========================
Variable Naming Conventions
===========================

While it is possible to set up the same OpenMDAO multiphysics problem with different sets of variable names, it is preferable for codes solving the same physics to use the same variable names to be more easily interchangeable.
This table provides the required names for coupling variables associated with a particular physics in MPhys.

The variable naming convention is defined as a Nested Classes with static variables.
The names are access from the ``MPhysVariables`` class.
If defining an component that uses the MPhys variable names repeatedly, it is often uses to define a local copy due to the long names that the nested variable classes.

.. code-block:: python

	import openmdao.api as om
	from mphys import MPhysVariables

	X_AERO0 = MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL
	U_AERO = MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS
	F_AERO = MPhysVariables.Aerodynamics.Surface.LOADS

	class AeroSolver(om.ImplicitComponent):
	    def setup(self):
	        self.add_input(X_AERO0, shape=5, distributed=True, tags=['mphys_coordinates'])
	        self.add_input(U_AERO0, shape=5, distributed=True, tags=['mphys_coupling'])
	        self.add_output(F_AERO0, shape=3, distributed=True, tags=['mphys_coupling'])

+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| Code Access                                                     | Associated Solver | MPhys tag         | Variable description                                             |
+=================================================================+===================+===================+==================================================================+
| MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES            | Aerodynamic       | mphys_coordinates | Aerodynamic surface coordinates from initial mesh file           |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT  | Aerodynamic       | mphys_coordinates | Aerodynamic surface coordinates (initial coordinates)            |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT | Aerodynamic       | mphys_coordinates | Aerodynamic surface coordinates (geometry-deform jig shape)      |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL         | Aerodynamic       | mphys_coordinates | Aerodynamic surface coordinates (jig shape)                      |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.COORDINATES_DEFORMED        | Aerodynamic       | mphys_coupling    | Aerodynamic surface coordinates (deformed)                       |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.DISPLACEMENTS               | Aerodynamic       | mphys_coupling    | Aerodynamic surface displacements                                |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.LOADS                       | Aerodynamic       | mphys_coupling    | Aerodynamic surface forces                                       |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.TEMPERATURE                 | Aerodynamic       | mphys_coupling    | Temperature for convective solver at interface                   |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.Surface.HEAT_FLOW                   | Aerodynamic       | mphys_coupling    | Convective heat flow at interface                                |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Structures.Mesh.COORDINATES                      | Structural        | mphys_coordinates | Structural coordinates from initial mesh file                    |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Structures.Geometry.COORDINATES_INPUT            | Structural        | mphys_coordinates | Structural coordinates (initial coordinates)                     |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Structures.Geometry.COORDINATES_OUTPUT           | Structural        | mphys_coordinates | Structural coordinates (geometry-deformed jig shape)             |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Structures.COORDINATES                           | Structural        | mphys_coordinates | Structural coordinates (jig shape)                               |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Structures.DISPLACEMENTS                         | Structural        | mphys_coupling    | Structural state vector (displacements)                          |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Structures.LOADS.AERODYNAMIC                     | Structural        | mphys_coupling    | Structural forces due to aerodynamics                            |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Thermal.Mesh.COORDINATES                         | Thermal           | mphys_coupling    | Thermal coordinates from initial mesh file                       |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Thermal.Geometry.COORDINATES_INPUT               | Thermal           | mphys_coupling    | Thermal coordinates (initial coordinates)                        |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Thermal.Geometry.COORDINATES_OUTPUT              | Thermal           | mphys_coupling    | Thermal coordinates (geometry-deformed jig shape)                |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Thermal.COORDINATES                              | Thermal           | mphys_coupling    | Thermal coordinates (jig shape)                                  |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Thermal.TEMPERATURE                              | Thermal           | mphys_coupling    | Temperature at interface (thermal solver side)                   |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+
| MPhysVariables.Thermal.HeatFlow.AERODYNAMIC                     | Thermal           | mphys_coupling    | heat flow at interface due to aerodynamics (thermal solver side) |
+-----------------------------------------------------------------+-------------------+-------------------+------------------------------------------------------------------+


To make swapping solvers easier, it is also helpful to share noncoupling variable names if possible:



+-------------------------------------------------------------+-------------------+-------------+---------------------------------------------------------------------+
| Variable                                                    | Associated Solver | MPhys tag   | Variable description                                                |
+=============================================================+===================+=============+=====================================================================+
| MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK  | Aerodynamic       | mphys_input | Angle of attack (please include units='deg' or 'rad' when declared) |
+-------------------------------------------------------------+-------------------+-------------+---------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.FlowConditions.YAW_ANGLE        | Aerodynamic       | mphys_input | Yaw angle (please include units='deg' or 'rad' when declared)       |
+-------------------------------------------------------------+-------------------+-------------+---------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER      | Aerodynamic       | mphys_input | Reference Mach number                                               |
+-------------------------------------------------------------+-------------------+-------------+---------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER  | Aerodynamic       | mphys_input | Reference Reynolds number                                           |
+-------------------------------------------------------------+-------------------+-------------+---------------------------------------------------------------------+
| MPhysVariables.Aerodynamics.FlowConditions.DYNAMIC_PRESSURE | Aerodynamic       | mphys_input | Dynamic pressure                                                    |
+-------------------------------------------------------------+-------------------+-------------+---------------------------------------------------------------------+
