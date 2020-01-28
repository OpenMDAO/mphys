*********************
OMFSI Model Hierarchy
*********************


=========
FSI Level
=========

The FSI level is the lowest level of the OMFSI hierarchy and contains the physics coupling in the MDAO problem.
That is it contains physics modules, such as an aerodynamic or structural solver, and potentially modules that transfer or interpolate between the physics modules, such as a load or displacement transfer schemes.
Modules added to this level by :ref:`assemblers` can be a single component or a group.

==============
Scenario Level
==============
The scenario level is an OpenMDAO group that represents a specific condition in a multipoint optimization.
For example, a scenario could be a cruise flight condition that requires a FSI group to determine the lift and drag.
The scenario group contains a FSI group and any scenario-specific computation that needs to occur before or after the associated FSI problem is solved.
For example, a sonic boom propagator requires the flow solution as an input but this one-way coupling does not require it to be in the FSI group; therefore, it should be put in the scenario group to be solved after the FSI group converges.

===========
Model Level
===========
The model level is the highest group level of the OpenMDAO model.
It can contain multiple scenario groups as well as any other computations that affect or are affected by multiple scenario groups.
An example of a component that computes before the scenarios would be a geometry engine that affects the shape of the bodies in all scenarios.
An example of a component that computes after the scenarios would be a cost function evaluation that averages the lift to drag ratio over a set of cruise scenarios that have different flight conditions.
