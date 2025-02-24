<img src="docs/logo/mphys_logo_no_background.png" alt= “MPhys” width="250">

[![Unit Tests and Docs](https://github.com/OpenMDAO/mphys/actions/workflows/unit_tests_and_docs.yml/badge.svg)](https://github.com/OpenMDAO/mphys/actions/workflows/unit_tests_and_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

MPhys is a package that standardizes high-fidelity multiphysics problems in OpenMDAO.
MPhys eases the problem set up, provides straightforward extension to new disciplines, and has a library of OpenMDAO groups for multidisciplinary problems addressed by its standard.

While MPhys does provide these conventions, it is not absolutely necessary to follow these guidelines in order to solve these types of problems with OpenMDAO given its very general coupling capability.
However, by following the MPhys conventions, the usage of OpenMDAO for multiphysics analysis will be modular across developer groups.
This eases technology transfer and collaboration in this area of research.
The standardization strives for modularity of multiphysics problems with large parallel physics codes.

## Install
To install the latest release version of mphys:
```bash
pip install mphys
```

For developers, clone the mphys repository then in the root directory do:
```bash
pip install -e .
```

## Documentation
Online documentation is available at [https://openmdao.github.io/mphys/](https://openmdao.github.io/mphys/).

### Building the Docs
The documentation includes N2 diagrams from the unit tests. Before building the docs, go into `tests/unit_tests` and run `python -m unittest`.
Then go into the `docs` directory and run `make html`.

# Citing MPhys
If you use MPhys in your research, please cite the [MPhys journal paper](https://link.springer.com/article/10.1007/s00158-024-03900-0).
A bibtex entry is provided [here](https://openmdao.github.io/mphys/references/citing_mphys.html).
A public version of the article is also available on [ResearchGate](https://www.researchgate.net/publication/387832759_MPhys_a_modular_multiphysics_library_for_coupled_simulation_and_adjoint_derivative_computation).

# Solvers compatible with MPhys
Open-source codes with builders and components compatible with mphys:

| Code                                                       | Recommended Version* | Analysis Type                  | Notes                                                                   |
|------------------------------------------------------------|----------------------|--------------------------------|-------------------------------------------------------------------------|
| [ADflow](https://github.com/mdolab/adflow)                 | 2.12.0               | Aerodynamics                   | Structured multi-block and overset CFD.                                 |
| [DAfoam](https://github.com/mdolab/dafoam)                 | 3.2.0                | Aerodynamics                   | Discrete Adjoint with OpenFOAM.                                         |
| [OpenAeroStruct](https://github.com/mdolab/openaerostruct) | 2.9.1                | Aerodynamics                   | Vortex lattice aerodynamics written using OpenMDAO.                     |
| [FunToFEM](https://github.com/smdogroup/funtofem)          | 0.3.8                | Load and Displacement Transfer | Point cloud based transfer scheme. Part of the FUNtoFEM package.        |
| [pyCycle](https://github.com/OpenMDAO/pyCycle)             | 4.3.0                | Propulsion                     | Thermodynamic cycle modeling library for engines.                       |
| [pyGeo](https://github.com/mdolab/pygeo)                   | 1.15.0               | Geometric Parameterization     | Wrapper for ESP, OpenVSP, and a free-form deformation parameterization. |
| [TACS](https://github.com/smdogroup/tacs)                  | 3.8.0                | Structures                     | Parallel Finite Element Analysis. |

\* Recommended version to run mphys examples. Older versions may still be supported.

# Examples
As noted their README.md files, some of the examples use codes that are not widely available;
however, they are still included in order to provide more illustrations of how mphys can be used.

# For developers

## Telecons

MPhys development is discussed in biweekly telecons that occur Mondays at 11AM Eastern Time.
If you would like to participate, contact Kevin Jacobson (kevin.e.jacobson@nasa.gov).

## Signed Commits
The MPhys `main` branch requires verified commits. See the instructions on how to sign commits [here](https://openmdao.org/newdocs/versions/latest/other_useful_docs/developer_docs/signing_commits.html).

## Tests
The test are written to use the testflo framework because it allows us to run tests with multiple cores.
To run the tests you will need to install testflo.

### Integration Tests
The integration tests check the interaction of mphys with several solvers.
These python packages are required to run them:
```
adflow
tacs
funtofem
testflo
paramerized
openaerostruct
```
and these input files. They can be obtained by running `get-input-files.sh`
```
wingbox.bdf
wing_vol_L1.cgns
wing_vol_L2.cgns
wing_vol_L3.cgns
ffd.xyz
```

to run the tests execute in the root directory
```bash
testflo -v tests
```

## Code Formatting
All pull requests automatically check for code formatting compliance using `flake8`, `black`, and `isort`.
Before submitting a PR check code changes adheres to this formating.
To run `flake8`, `black`, and `isort` locally, use the folowing commands:
```commandline
$ pip install flake8 black isort
$ wget https://raw.githubusercontent.com/mdolab/.github/main/.flake8 -O .flake8_mdolab  # download flake8 configuration for mdolab
$ python -m flake8 . --append-config .flake8_mdolab --count --show-source --statistics
$ python -m black . --check --diff
$ python -m isort . --check-only --diff
```

# Software Assurance Plan

MPhys has been deemed as Class-E software, according to the [7120.5D Specification](https://www.nasa.gov/pdf/423715main_NPR_7120-5_HB_FINAL-02-25-10.pdf).
To maintain software quality and assure functionality, MPhys includes a unit and integration test suite.
Before any pull requests are merged, all of those tests must pass.
The tests are run as part of a continuous integration system, automatically upon pull request submission.

We require all commits to be signed to ensure that we know the "identity" (at least that the commit is actually coming from the account it claims to be).
Unsigned commits will not be accepted.

The Bandit static analysis tool is run on the codebase to check for any "simple" security issues.
This checks for basic vulnerabilities like having API keys, user names, or passwords in the repository.
Bandit is run manually on the repository before any major releases are made.
