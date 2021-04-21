******************************
Extending the Scenario Library
******************************



.. _dev_coupling_group:

===============
Coupling Groups
===============


.. automodule:: mphys.coupling_group

.. autoclass:: CouplingGroup
    :members:

.. _dev_scenario_group:

=========
Scenarios
=========
Your custom Scenario should at least implement the ``initialize`` and ``setup`` methods.

----------
Initialize
----------
The Scenario's ``initialize`` method should declare the necessary builders as options.
An ``in_MultipointParallel`` option should also be included if that mode is implemented.
Otherwise, the developer free to

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
