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
```
adflow
tacs
funtofem
testflo
```

to run the tests execute
```bash
testflo -v tests
```
