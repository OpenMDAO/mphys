%%%%%%%%%%%%%%%%%%%%%%%
Aerostructural Scenario
%%%%%%%%%%%%%%%%%%%%%%%

The :class:`ScenarioAeroStructural<mphys.scenario_aerostructural.ScenarioAeroStructural>` is for static fluid structure interaction problems.
The primary physics modules required for this problem are:
1. The aerodynamics which computes forces given the displaced aerodynamic surface coordinates.
2. The structures which computes structural displacements given the loads at structural nodes.
3. The displacement transfer which projects the structural displacements to the aerodynamic surface mesh
4. The load transfer which computes the loads on the structure from the aerodynamic output.

MPhys will add a :class:`~mphy.geo_disp.GeoDisp` subsystem to compute the displaced aerodynamic coordinates given the undeformed surface coordinates and the displacements.

Builder Requirements
====================

Load and Displacement Transfer Builder
--------------------------------------
Because the load and displacement transfers are typically tied together by the principle of virtual work, but are not adjacent in the coupling loop,
the load and displacement Builder's :meth:`~Builder.get_coupling_group_subsystem` must return both the displacement transfer and load tranfer subsystems as a tuple.

Structural Solver Builder
-------------------------
The structural solver builder must implement the ``get_ndof`` method in order for the displacement transfer to know if it needs to slice the displacements from the full structural state vector.
For example, the structural state vector for linear shell elements includes linearized rotation degrees of freedom at each node in addition to the translational displacements.

Default Solvers
===============
The default solvers are NonlinearBlockGS and LinearBlockGS with ``use_aitken=True``.

Options
=======
.. embed-options::
  mphys.scenario_aerostructural
  ScenarioAeroStructural
  options

N2: Basic
=========

.. embed-pregen-n2::
  ../tests/unit_tests/n2/TestScenarioAeroStructural.html

N2: in_MultipointParallel
=========================

.. embed-pregen-n2::
  ../tests/unit_tests/n2/TestScenarioAeroStructuralParallel.html

N2: in_MultipointParallel with geometry_builder
===============================================

.. embed-pregen-n2::
  ../tests/unit_tests/n2/TestScenarioAeroStructuralParallelWithGeometry.html
