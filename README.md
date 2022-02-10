<<<<<<< HEAD
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
=======
[![Build Status](https://github.com/mir-group/flare_pp/actions/workflows/main.yml/badge.svg)](https://github.com/mir-group/flare_pp/actions)

# flare++
Major features:
* Bayesian force fields based on sparse Gaussian process regression.
* Multielement many-body descriptors based on the [atomic cluster expansion](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104).
* Mapping to efficient parametric models.
* Coupling to [LAMMPS](https://www.lammps.org/) for large-scale molecular dynamics simulations.

Check out our preprint introducing flare++ [here](https://arxiv.org/abs/2106.01949).

## Demo and Instructions for Use
An introductory tutorial in Google Colab is available [here](https://colab.research.google.com/drive/18_pTcWM19AUiksaRyCgg9BCpVyw744xv). The tutorial shows how to run flare++ on energy and force data, demoing "offline" training on the MD17 dataset and "online" on-the-fly training of a simple aluminum force field. A video walkthrough of the tutorial, including detailed discussion of expected outputs, is available [here](https://youtu.be/-FH_VqRQrso).

The tutorial takes a few minutes to run on a normal desktop computer or laptop (excluding installation time).

## Installation guide
### Pip installation
The easiest way to install flare++ is with pip. Just run the following command:
```
pip install flare_pp
```
This will take a few minutes on a normal desktop computer or laptop.

If you're installing on Harvard's compute cluster, make sure to load the following modules first:
```
module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01
```

### Developer's installation guide
For developers, please check the [installation guide](https://mir-group.github.io/flare_pp/installation.html).

### Compiling LAMMPS
See [lammps_plugins/README.md](https://github.com/mir-group/flare_pp/blob/master/lammps_plugins/README.md).

### Trouble shooting
If you have problem compiling and installing the code, please check the [FAQs](https://mir-group.github.io/flare_pp/faqs.html) to see if your problem is covered. Otherwise, please open an issue or contact us.

## System requirements
### Software dependencies
* GCC 9
* Python 3
* pip>=20

MKL is recommended but not required. All other software dependencies are taken care of by pip.

The code is built and tested with Github Actions using the GCC 9 compiler. (You can find a summary of recent builds [here](https://github.com/mir-group/flare_pp/actions).) Other C++ compilers may work, but we can't guarantee this.

### Operating systems
flare++ is tested on a Linux operating system (Ubuntu 20.04.3), but should also be compatible with Mac and Windows operating systems. If you run into issues running the code on Mac or Windows, please post to the [issue board](https://github.com/mir-group/flare_pp/issues).

### Hardware requirements
There are no non-standard hardware requirements to download the software and train simple models&mdash;the introductory tutorial can be run on a single cpu. To train large models (10k+ sparse environments), we recommend using a compute node with at least 100GB of RAM.

## Documentation
Preliminary documentation of the C++ source code can be accessed [here](https://mir-group.github.io/flare_pp/). 
>>>>>>> pp/dev
