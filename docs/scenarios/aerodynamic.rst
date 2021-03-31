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
  mphys.scenario_aerodynamic
  ScenarioAerodynamic
  options


N2:Basic
========

.. embed-pregen-n2::
  ../tests/unit_tests/n2/TestScenarioAerodynamic.html


N2: in_MultipointParallel
=========================

.. embed-pregen-n2::
  ../tests/unit_tests/n2/TestScenarioAerodynamicParallel.html


N2: in_MultipointParallel with geometry_builder
===============================================

.. embed-pregen-n2::
  ../tests/unit_tests/n2/TestScenarioAerodynamicParallelWithGeometry.html
