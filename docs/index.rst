%%%%%%%%%%%%%%%%%%%%%%%
Documentation for MPhys
%%%%%%%%%%%%%%%%%%%%%%%

MPhys is a package that standardizes high-fidelity multiphysics problems in OpenMDAO.
MPhys eases the problem set up, provides straightforward extension to new disciplines, and has a library of OpenMDAO groups for multidisciplinary problems addressed by its standard.

While MPhys does provide these conventions, it is not absolutely necessary to follow these guidelines in order to solve these types of problems with OpenMDAO given its very general coupling capability.
However, by following the MPhys conventions, the usage of OpenMDAO for multiphysics analysis will be modular across developer groups.
This eases technology transfer and collaboration in this area of research.
The standardization strives for modularity of multiphysics problems with large parallel physics codes.

MPhys Basics
************

These are descriptions of how MPhys works and how it interfaces with solvers and OpenMDAO.

.. toctree::
  :maxdepth: 1
  :caption: MPhys Basics

  basics/model_hierarchy.rst
  basics/tagged_promotion.rst
  basics/builders.rst
  basics/naming_conventions.rst

.. _scenario_library:

**********************
MPhys Scenario Library
**********************

These are descriptions of the groups in the MPhys library of multiphysics problems.
They describe physics problem being solved, the standards set by MPhys, requirements of the Builders, and the options available for each group.

.. toctree::
  :maxdepth: 1
  :caption: Multiphysics Scenarios

  scenarios/aerostructural.rst

.. toctree::
  :maxdepth: 1
  :caption: Single Discipline Scenarios

  scenarios/structural.rst
  scenarios/aerodynamic.rst

**********************
MPhys Developers Guide
**********************

These pages provide more details of how MPhys works and how to add to the MPhys scenario library.

.. toctree::
  :maxdepth: 1
  :caption: Developers Guide

  developers/mphys_group.rst
  developers/new_multiphysics_problems.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
