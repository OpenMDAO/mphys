******************************
Extending the Scenario Library
******************************



.. _dev_coupling_group:

===============
Coupling Groups
===============

In ``initialize``, the coupling group needs to have the required builders declared as options.
In the setup phase, :func:`~mphys.builder.Builder.get_coupling_group_subsystem` is used to get subsystems from the builders,
and ``self.mphys_add_subsystem`` is used to add them.
Other components like balance components can be added directly.
Setting a default linear and nonlinear solver suitable for this type of problem is also helpful in ``setup``.
In most cases using a configure phase is not necessary for the scenario, but if you do implement a ``configure``
method, you must call ``super().configure()`` to do the tagged promotion of variables from the scenario's subsystems.

.. automodule:: mphys.coupling_group

.. autoclass:: CouplingGroup
    :members:

.. _dev_scenario_group:

=========
Scenarios
=========
Your custom Scenario should at least implement the ``initialize`` and ``setup`` methods.
As with the ``CouplingGroup``, you must call the ``configure`` method of the parent class if you implement a
``configure`` method in the Scenario.

----------
Initialize
----------
The Scenario's ``initialize`` method should declare the necessary builders as options.
An ``in_MultipointParallel`` option should also be included if that mode of operation will be implemented.
Otherwise the developer is free to add options specific to the scenario type.

-----
Setup
-----
The basic pattern for the scenario group's setup method is:

0. If ``in_MultipointParallel``, initialize all the builders
1. Call :func:`~mphys.scenario.Scenario.mphys_add_pre_coupling_subsystem` for each builder.
2. Instantiate the associated :class:`~mphys.coupling_group.CouplingGroup`.
3. Call :func:`~mphys.scenario.Scenario.mphys_add_post_coupling_subsystem` for each builder.


.. automodule:: mphys.scenario

.. autoclass:: Scenario
    :members:
