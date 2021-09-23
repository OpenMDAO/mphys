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

## Building the Docs
The documentation includes N2 diagrams from the unit tests. Before building the docs, go into `tests/unit_tests` and run `python -m unittest`.
Then go into the `docs` directory and run `make html`.


## Tests
The test are written to use the testflo framework because it allows us to run tests with multiple cores.
To run the tests you will need.

### Integration Tests
These python packages
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
wing_vol_L3.cgns
ffd.xyz
```

to run the tests execute in the root directory
```bash
testflo -v tests
```
