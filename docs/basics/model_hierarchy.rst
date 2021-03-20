***************
Model Hierarchy
***************

Mphys uses a pattern to build multiphysics optimization problems.
Each level of the pattern is a different type of group that Mphys provides.

The highest level of the model is the multipoint group.
The multipoint group consist of scenarios which represent different conditions and/or types of analyses to performed.
Within the scenario is the coupling group represents the primary multiphysics problem for the scenario.


:ref:`builders` are used to help populate these levels of the model hierarchy with subsystems from the solvers.
:ref:`tagged_promotion` is used to promote specific variables to the level of scenario.

=============
CouplingGroup
=============

The CouplingGroup is the primary physics coupling being solved.
That is it contains physics modules, such as an aerodynamic or structural solver, and potentially modules that transfer or interpolate between the physics modules, such as a load or displacement transfer schemes.
Each type of scenario typically has an associated coupling group that it will add.
The scenario-specific coupling group will have a default nonlinear and linear solver associated with,
but these can be overwritten with the optional arguments to the Multipoint group's `mphys_add_scenario` method.

========
Scenario
========
The scenario level is an OpenMDAO group that represents a specific condition in a multipoint optimization.
For example, a scenario could be a cruise flight condition that requires a coupling group to determine the lift and drag.
The scenario group contains a coupling group and any scenario-specific computation that needs to occur before or after the associated coupled problem is solved.
For example, a sonic boom propagator requires the flow solution as an input but this one-way coupling does not require it to be in the coupling group; therefore, it should be put in the scenario group to be solved after the coupling group converges.

==========
Multipoint
==========
