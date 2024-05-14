%%%%%%%%%%%%%%%%%%%%%%%
Aeropropulsive Scenario
%%%%%%%%%%%%%%%%%%%%%%%

The :class: `ScenarioAeropropulsive <mphys.scenario_aeropropulsive.ScenarioAeropropulsive>` is for static coupled aerodynamic and propulsion problems.
The primary physics modules required for this problem are:
1. The aerodynamics which computes forces and intensive thermodynamic properties given the aerodynamic surface coordinates.
2. The thermodynamic cycle model which computes the effect of the propulsion system on the flowfield.
3. The boundary condition (BC) coupling that enforces the coupling between the aerodynamic and propulsion systems.

Builder Requirements
====================

Propulsion Solver Builder
-------------------------
The propulsion builder constructs the thermodynamic cycle model(s) that are used to compute the effect of the propulsion system on the aerodynamics.

BC Coupling Builder 
--------------------
The BC coupling builder implements consistency constraints formulated as residuals between the aerodynamic and propulsion disciplines.
The consistency constraints enforce the aeropropulsive coupling through the optimization problem.

Options
=======
.. embed-options::
  mphys.scenario_aeropropulsive
  ScenarioAeropropulsive
  options