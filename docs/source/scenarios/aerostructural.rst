%%%%%%%%%%%%%%%%%%%%%%%
Aerostructural Scenario
%%%%%%%%%%%%%%%%%%%%%%%

The :class:`ScenarioStructural<mphys.scenario_structural.ScenarioStructural>` is for static fluid structure interaction problems.

N2 Diagram
==========

Builder Requirements
====================

Load and Displacement Transfer Builder
--------------------------------------
Because the load and displacement transfers are typically tied together by the principle of virtual work, but are not adjacent in the coupling loop,
the load and displacement Builder's :meth:`Builder.get_coupling_group_subsystem` must return both the displacement transfer and load tranfer subsystems as a tuple.

Structural Solver Builder
-------------------------
The structural solver builder must implement the `get_ndof` method in order for the displacement transfer to know if it needs to slice the displacements from the full state vector.

Default Solvers
===============
The default solvers are NonlinearRunOnce and LinearRunOnce that execute the pre coupling, coupling, and post coupling subsystems in order.

Options
=======
.. embed-options::
  mphys.scenario_structural
  ScenarioStructural
  options
