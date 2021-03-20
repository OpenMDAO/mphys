%%%%%%%%%%%%%%%%%%%%%%%
Documentation for Mphys
%%%%%%%%%%%%%%%%%%%%%%%

Mphys is a package that standardizes high-fidelity multiphysics problems in OpenMDAO.
Mphys eases the problem set up, provides straightforward extension to new disciplines, and has a library of OpenMDAO groups for multidisciplinary problems addressed by its standard.

While Mphys does provide these conventions, it is not absolutely necessary to follow these guidelines in order to solve these types of problems with OpenMDAO given its very general coupling capability.
However, by following the Mphys conventions, the usage of OpenMDAO for multiphysics analysis will be modular across developer groups.
This eases technology transfer and collaboration in this area of research.
The standardization strives for modularity of multiphysics problems with large parallel physics codes.

Mphys Basics
************

These are descriptions of how Mphys works and how it interfaces with solvers and OpenMDAO.

.. toctree::
  :maxdepth: 1
  :caption: Mphys Basics

  basics/model_hierarchy.rst
  basics/tagged_promotion.rst
  basics/builders.rst
  basics/naming_conventions.rst

Mphys Scenario Library
**********************

These are descriptions of the groups in the Mphys library of multiphysics problems.
They describe physics problem being solved, the standards set by Mphys, requirements of the Builders, and the options available for each group.

.. toctree::
  :maxdepth: 1
  :caption: Multiphysics Scenarios

  scenarios/aerostructural.rst

.. toctree::
  :maxdepth: 1
  :caption: Single Discipline Scenarios

  scenarios/structural.rst
  scenarios/aerodynamic.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
