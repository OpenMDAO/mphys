This example performs an aerostructural optimization using a TACS structural analysis coupled to an OpenAerostruct aerodynamic analysis.
The optimization model is based off a Boeing 777 class aircraft geometry (publicly available [here](https://hangar.openvsp.org/vspfiles/375)).
The goal of the optimization is to minimize the wing structural weight under a cruise flight condition subject to an aggregated failure constraint.

The optimization can be run from the `run_oas_tacs_wing.py` script.
In order to run this script the user is required to [OpenVSP Python API](https://openvsp.org/api_docs/latest/) installed, 
in addition to the standard MPhys libraries (TACS, OpenAeroStruct, FunToFEM, etc.).