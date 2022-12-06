**************
The MphysGroup
**************

The purpose of the MphysGroup is to implement the mechananics of promoting the MPhys variables by tags.
Subsystems with tagged variables that will be promoted are added with the :func:`~mphys.mphys_group.MphysGroup.mphys_add_subsystem` method.
Subsystems that have variables that should not be promoted can still be added with ``add_subsystem``
The automated promotion of tagged variables is done during the configure phase of OpenMDAO setup.
If you need to use ``configure`` in your CouplingGroup or Scenario group, be sure to call the parent's configure with
``super().configure()``.

The :class:`~mphys.mphys_group.MphysGroup` is the base class of the :ref:`dev_coupling_group` and :ref:`dev_scenario_group`.
While it is important to understand the MphysGroup's :func:`~mphys.mphys_group.MphysGroup.configure` and :func:`~mphys.mphys_group.MphysGroup.mphys_add_subsystem` interactions,
any new scenario or coupling group should inherit from :class:`~mphys.scenario.Scenario` and :class:`~mphys.coupling_group.CouplingGroup`.
rather than subclassing MphysGroup directly.

.. automodule:: mphys.mphys_group

.. autoclass:: MphysGroup
    :members:


==============================
Manual Connection of Variables
==============================

In some instances, the use of automated promotion is not appropriate.
Because the MphysGroup inherits from the standard OpenMDAO group,
subsystems can be added with the standard ``add_subsystem`` method and connected manually.
