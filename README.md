# MPHYS
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
