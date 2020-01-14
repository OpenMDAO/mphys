%%%%%%%%%%%%%%%%%%%%%%%
Documentation for OMFSI
%%%%%%%%%%%%%%%%%%%%%%%

OMFSI is a set of standards and naming conventions for high-fidelity multiphysics problems with OpenMDAO.
The standardization strives for modularity of multiphysics problems with large parallel physics codes.
The roots of OMFSI are in aerostructural optimization, but these practices can be extends to other physics as well.

While OMFSI does provide these conventions, it is not absolutely necessary to follow these guidelines in order to solve fluid-structure interaction problems with OpenMDAO given its very general coupling capability.
However, by following a standard set of variable names and model development conventions, the usage of OpenMDAO for fluid-structure interaction analysis will be modular across groups.
This eases technology transfer and collaboration in this area of research.

.. toctree::
  :maxdepth: 1
  :caption: Contents:
  :name: developersguide

  developers/model_hierarchy.rst
  developers/assemblers.rst
  developers/omfsi_conventions.rst
  developers/parallelism.rst
  developers/aerodynamic_solvers.rst
  developers/structural_solvers.rst
  developers/load_and_displacement_transfers.rst

.. toctree::
  :maxdepth: 1
  :name: examples

  examples/fsi_cfd_fem.rst
  examples/fsi_vlm_fem.rst
  examples/fsi_cfd_modal.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
