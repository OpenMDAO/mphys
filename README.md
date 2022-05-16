# MPHYS
[![Unit Tests and Docs](https://github.com/OpenMDAO/mphys/actions/workflows/unit_tests_and_docs.yml/badge.svg)](https://github.com/OpenMDAO/mphys/actions/workflows/unit_tests_and_docs.yml)

MPHYS is a framework for coupling high-fidelity physics though OpenMDAO


## Install
Because MPHYS is written in pure python to install simply execute the following in the MPHYS root directory
```bash
pip install .
```
or for a development version
```bash
pip install -e .
```

## Documentation
Online documentation is available at [https://openmdao.github.io/mphys/](https://openmdao.github.io/mphys/).

### Building the Docs
The documentation includes N2 diagrams from the unit tests. Before building the docs, go into `tests/unit_tests` and run `python -m unittest`.
Then go into the `docs` directory and run `make html`.

# Solvers compatible with mphys
Open-source codes with builders and components compatible with mphys:

| Code                                                       | Analysis Type                  | Notes                                                                                              |
|------------------------------------------------------------|--------------------------------|----------------------------------------------------------------------------------------------------|
|[ADflow](https://github.com/mdolab/adflow)                  | Aerodynamics                   | Structured multi-block and overset CFD.                                                            |
|[DAfoam](https://github.com/mdolab/dafoam)                  | Aerodynamics                   | Discrete Adjoint with OpenFOAM.                                                                    |
|[MELD](https://github.com/smdogroup/funtofem)               | Load and Displacement Transfer | Point cloud based transfer scheme. Part of the FUNtoFEM package.                                   |
|[pyCycle](https://github.com/OpenMDAO/pyCycle)              | Propulsion                     | Thermodynamic cycle modeling library for engines.                                                  |
|[pyGeo](https://github.com/mdolab/pygeo)                    | Geometric Parameterization     | Wrapper for ESP, OpenVSP, and a free-form deformation parameterization.                            |
|[TACS](https://github.com/smdogroup/tacs)                   | Structures                     | Parallel Finite Element Analysis.                                                                  |

# Examples
As noted their README.md files, some of the examples use codes that are not widely available;
however, they are still included in order to provide more illustrations of how mphys can be used.

# For developers

## Signed Commits
The mphys `main` branch requires verified commits. See the instructions on how to sign commits [here](https://openmdao.org/newdocs/versions/latest/other_useful_docs/developer_docs/signing_commits.html).

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

# Software Assurance Plan

Mphys has been deemed as Class-E software, according to the [7120.5D Specification](https://www.nasa.gov/pdf/423715main_NPR_7120-5_HB_FINAL-02-25-10.pdf).
To maintain software quality and assure functionality, Mphys includes a unit and integration test suite.
Before any pull requests are merged, all of those tests must pass.
The tests are run as part of a continuous integration system, automatically upon pull request submission.

We require all commits to be signed to ensure that we know the "identity" (at least that the commit is actually coming from the account it claims to be).
Unsigned commits will not be accepted.

The Bandit static analysis tool is run on the codebase to check for any "simple" security issues.
This checks for basic vulnerabilities like having API keys, user names, or passwords in the repository.
Bandit is run manually on the repository before any major releases are made.
