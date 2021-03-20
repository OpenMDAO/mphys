.. _tagged_promotion:

****************
Tagged Promotion
****************

Mphys uses tags to selectively promote variables to the level of scenario.
There are four types of tags in Mphys that represent different types of data.

* ``mphys_coupling``: Coupling variables are states that need to be shared among subsystems in the scenario.  Both inputs and outputs with this tag are promoted.
* ``mphys_coordinates``: These are the coordinates of the mesh of a particular discipline. Both inputs and outputs with this tag are promoted.
* ``mphys_input``: Inputs are any variable like design variables that come from outside the scenario. The user will need to manually connect the source to ``{scenario_name}.{input_name}``. Only inputs with this tag are promoted.

* ``mphys_result``: Results are any other output from components that you wish to promote to the level of the scenario. The ``scenario_name.result_name`` can be accessed as an objective or constraint, or they can be connected to other subsystems for further computation. Only outputs with this tag are promoted.


==========
MphysGroup
==========

The :class:`~mphys.mphys_group.MphysGroup` is the base class of the :ref:`coupling_groups` and :ref:`scenario_groups`.
Subsystems with tagged variables that will be promoted are added with the ``mphys_add_subsystem`` method.
Subsystems that have variables that should not be promoted can still be added with ``add_subsystem``
The ``configure`` method of the MphysGroup automates the promotion tagged variables.

.. automodule:: mphys.mphys_group

.. autoclass:: MphysGroup
  :members:
