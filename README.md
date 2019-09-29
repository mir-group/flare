[![Build Status](https://travis-ci.org/mir-group/flare.svg?branch=master)](https://travis-ci.org/mir-group/flare) [![documentation](https://readthedocs.org/projects/flare/badge/?version=latest)](https://readthedocs.org/projects/flare) [![codecov](https://codecov.io/gh/mir-group/flare/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-group/flare)

# FLARE: Fast Learning of Atomistic Rare Events

FLARE is an open-source Python package for creating fast and accurate atomistic potentials. Documentation of the code is in progress, and can be accessed here: https://flare.readthedocs.io/


## Prerequisites
1. To train a potential on the fly, you need a working installation of [Quantum ESPRESSO](https://www.quantum-espresso.org) or [CP2K](https://www.cp2k.org).
2. FLARE requires Python 3 with the packages specified in `requirements.txt`. This is taken care of by `pip`.

## Installation
Install FLARE in two ways:
1. Download and install automatically:
    ```
        pip install git+https://github.com/mir-group/flare.git
    ```
2. Download this repository and install (required for unit tests):
    ```
        git clone https://github.com/mir-group/flare
        cd flare
        pip install .
    ```


## Tests
We recommend running unit tests to confirm that FLARE is running properly on your machine. We have implemented our tests using the pytest suite. You can call `pytest` from the command line in the tests directory to validate that you can call Quantum ESPRESSO and that your Numba installation is being correctly used by FLARE.

Instructions (either DFT package will suffice):
```
pip install pytest
cd tests
PWSCF_COMMAND=/path/to/pw.x CP2K_COMMAND=/path/to/cp2k pytest
```

## References
[1] Jonathan Vandermause, Steven B. Torrisi, Simon Batzner, Alexie M. Kolpak, and Boris Kozinsky. *On-the-fly Bayesian active learning of interpretable force fields for atomistic rare events.* https://arxiv.org/abs/1904.02042
