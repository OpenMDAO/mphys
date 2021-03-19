%%%%%%%%%%%%%%%%%%%%
Aerodynamic Scenario
%%%%%%%%%%%%%%%%%%%%
The :class:`ScenarioAerodynamic<mphys.scenario_aerodynamic.ScenarioAerodynamic>` is for scenarios that only need the aerodynamic solver.

Default Solvers
===============
The default solvers are NonlinearRunOnce and LinearRunOnce that execute the pre coupling, coupling, and post coupling subsystems in order.

StructuralScenario Options
==========================
.. embed-options::
  mphys.scenario_structural
  ScenarioStructural
  options
