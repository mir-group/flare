[![Build Status](https://github.com/mir-group/flare/actions/workflows/main.yml/badge.svg)](https://github.com/mir-group/flare/actions) [![documentation](https://readthedocs.org/projects/flare/badge/?version=latest)](https://readthedocs.org/projects/flare) [![pypi](https://img.shields.io/pypi/v/mir-flare)](https://pypi.org/project/mir-flare/) [![activity](https://img.shields.io/github/commit-activity/m/mir-group/flare)](https://github.com/mir-group/flare/commits/master) [![codecov](https://codecov.io/gh/mir-group/flare/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-group/flare)

# FLARE: Fast Learning of Atomistic Rare Events

<p align="center">
  <img width="659" height="303" src="https://github.com/mir-group/flare/blob/master/docs/images/Flare_logo.png?raw=true">
</p>

FLARE is an open-source Python package for creating fast and accurate interatomic potentials. Documentation of the code can be accessed here: https://flare.readthedocs.io/

We have an introductory tutorial in Google Colab available [here](https://colab.research.google.com/drive/1Q2NCCQWYQdTW9-e35v1W-mBlWTiQ4zfT).

## Major Features

* Gaussian Process Force Fields
  * 2- and 3-body multi-element kernels
  * Maximum likelihood hyperparameter optimization

* On-the-Fly Training
  * Coupling to Quantum Espresso, CP2K, and VASP DFT engines

* Mapped Gaussian Processes
  * Mapping to efficient cubic spline models

* ASE Interface
  * ASE calculator for GP models
  * On-the-fly training with ASE MD engines

* Module for training GPs from AIMD trajectories

## Many-body extension

If you want to train a more accurate many-body potential on the fly, check out our many-body extension [flare++](https://github.com/mir-group/flare_pp), with an introductory tutorial available [here](https://colab.research.google.com/drive/18_pTcWM19AUiksaRyCgg9BCpVyw744xv).

## Prerequisites
1. To train a potential on the fly, you need a working installation of a DFT code compatible with ASE (e.g. [Quantum ESPRESSO](https://www.quantum-espresso.org), [CP2K](https://www.cp2k.org), or [VASP](https://www.vasp.at/)).
2. FLARE requires Python 3 with the packages specified in `requirements.txt`. This is taken care of by `pip`.

## Installation
FLARE can be installed in two different ways.
1. Download and install automatically:
    ```
    pip install mir-flare
    ```
2. Download this repository and install (required for unit tests):
    ```
    git clone https://github.com/mir-group/flare
    cd flare
    pip install .
    ```


## Tests
We recommend running unit tests to confirm that FLARE is running properly on your machine. We have implemented our tests using the pytest suite. You can call `pytest` from the command line in the tests directory to validate that Quantum ESPRESSO or CP2K are working correctly with FLARE.

Instructions (either DFT package will suffice):
```
pip install pytest
cd tests
PWSCF_COMMAND=/path/to/pw.x CP2K_COMMAND=/path/to/cp2k pytest
```

## References
- If you use FLARE in your research, or any part of this repo (such as the GP implementation), please cite the following paper:

  [1] Vandermause, J., Torrisi, S. B., Batzner, S., Xie, Y., Sun, L., Kolpak, A. M. & Kozinsky, B. *On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events.* npj Comput Mater 6, 20 (2020). https://doi.org/10.1038/s41524-020-0283-z

- If you use MGP or LAMMPS pair style, please cite the following paper:

  [2] Xie, Y., Vandermause, J., Sun, L. et al. *Bayesian force fields from active learning for simulation of inter-dimensional transformation of stanene.* npj Comput Mater 7, 40 (2021). https://doi.org/10.1038/s41524-021-00510-y
