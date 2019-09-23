[![Build Status](https://travis-ci.org/mir-group/flare.svg?branch=master)](https://travis-ci.org/mir-group/flare) [![codecov](https://codecov.io/gh/mir-group/flare/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-group/flare)

# FLARE: Fast Learning of Atomistic Rare Events

FLARE is an open-source Python package for creating fast and accurate atomistic potentials. Documentation of the code is in progress, and can be accessed here: https://flare.readthedocs.io/


## Prerequisites
1. To train a potential on the fly, you'll need to have a working installation of Quantum ESPRESSO on your machine. The instructions for installation can be found here: https://www.quantum-espresso.org/
2. Our kernels and environment objects require the Python package numba. If you're using Anaconda, you can get it with the command `conda install numba`.
3. In order for unit testing to work:<br/>
   a. set an environment variable called "PWSCF_COMMAND" to point to your pw.x Quantum ESPRESSO binary.<br/>
   b. ensure you have the pytest python package installed.
4. Add the flare directory to your Python path.

## Tests
We recommend running unit tests to confirm that FLARE is running properly on your machine. We have implemented our tests using the pytest suite. You can call 'pytest' from the command line in the tests directory to validate that you can call Quantum ESPRESSO and that your Numba installation is being correctly used by FLARE.

## References
[1] Jonathan Vandermause, Steven B. Torrisi, Simon Batzner, Alexie M. Kolpak, and Boris Kozinsky. *On-the-fly Bayesian active learning of interpretable force fields for atomistic rare events.* https://arxiv.org/abs/1904.02042
