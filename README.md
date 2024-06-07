[![Build Status](https://github.com/mir-group/flare/actions/workflows/flare.yml/badge.svg)](https://github.com/mir-group/flare/actions) [![pypi](https://img.shields.io/pypi/v/mir-flare)](https://pypi.org/project/mir-flare/) [![activity](https://img.shields.io/github/commit-activity/m/mir-group/flare)](https://github.com/mir-group/flare/commits/master) [![codecov](https://codecov.io/gh/mir-group/flare/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-group/flare)

***NOTE: This is the latest release [1.3.3](https://github.com/mir-group/flare/releases/tag/1.3.3) which includes significant changes compared to the previous version [0.2.4](https://github.com/mir-group/flare/releases/tag/0.2.4). Please check the updated tutorials and documentations from the links below.***

# FLARE: Fast Learning of Atomistic Rare Events

<p align="center">
  <img width="527" height="242" src="https://github.com/mir-group/flare/blob/master/docs/images/Flare_logo.png?raw=true">
</p>

FLARE is an open-source Python package for creating fast and accurate interatomic potentials.

## Major Features

<p align="center">
  <img src="https://github.com/mir-group/flare/blob/development/docs/images/Flare_features.jpg?raw=true">
</p>

Note:

We implement Sparse GP, all the kernels and descriptors in C++ with Python interface.

We implement Full GP, Mapped GP, RBCM, Squared Exponential kernel and 2+3-body descriptors in Python.

Please do NOT mix them.

## Documentations and Tutorials

Documentation of the code can be accessed here: https://mir-group.github.io/flare

[Applications using FLARE and gallery](https://mir-group.github.io/flare/related.html)

### Google Colab Tutorials

[FLARE (ACE descriptors + sparse GP)](https://colab.research.google.com/drive/1rZ-p3kN5CJbPJgD8HuQHSc7ecmwZYse6).
The tutorial shows how to run flare with ACE and SGP on energy and force data, demoing "offline" training on the MD17 dataset and "online" on-the-fly training of a simple aluminum force field. All the trainings use yaml files for configuration.

[FLARE (LAMMPS active learning)](https://bit.ly/flarelmpotf)
This tutorial demonstrates new functionality for running active learning all within LAMMPS, with LAMMPS running the dynamics to allow arbitrarily complex molecular dynamics workflows while maintaining a simple interface. This also demonstrates how to use the C++ API directly from Python through `pybind11`. Finally, there's a simple demonstration of phonon calculations with FLARE using `phonopy`.

[FLARE (ACE descriptors + sparse GP) with LAMMPS](https://colab.research.google.com/drive/1qgGlfu1BlXQgSrnolS4c4AYeZ-2TaX5Y).
The tutorial shows how to compile LAMMPS with FLARE pair style and uncertainty compute code, and use LAMMPS for Bayesian active learning and uncertainty-aware molecular dynamics.

[Compute thermal conductivity from FLARE and Boltzmann transport equations](https://phoebe.readthedocs.io/en/develop/tutorials/mlPhononTransport.html).
The tutorial shows how to use FLARE (LAMMPS) potential to compute lattice thermal conductivity from Boltzmann transport equation method, with [Phono3py](https://phonopy.github.io/phono3py/) for force constants calculations and [Phoebe](https://mir-group.github.io/phoebe/) for thermal conductivities.

[Using your own customized descriptors with FLARE](https://colab.research.google.com/drive/1VzbIPmx1z-uygKstOYTj2Nqr53AMC5NL?usp=sharing).
The tutorial shows how to attach your own descriptors with FLARE sparse GP model and do training and testing.

All the tutorials take a few minutes to run on a normal desktop computer or laptop (excluding installation time).

## Installation
### Pip installation
Please check the [installation guide here](https://mir-group.github.io/flare/installation/install.html).
This will take a few minutes on a normal desktop computer or laptop.

### Developer's installation guide
For developers, please check the [installation guide](https://mir-group.github.io/flare/installation/install.html#developer-s-installation-guide).

### Compiling LAMMPS
See [documentation on compiling LAMMPS with FLARE](https://mir-group.github.io/flare/installation/lammps.html)

### Trouble shooting
If you have problem compiling and installing the code, please check the [FAQs](https://mir-group.github.io/flare/installation/install.html#trouble-shooting) to see if your problem is covered. Otherwise, please open an issue or contact us.

## System requirements
### Software dependencies
* GCC 9
* Python 3
* pip>=20

MKL is recommended but not required. All other software dependencies are taken care of by pip.

The code is built and tested with Github Actions using the GCC 9 compiler. (You can find a summary of recent builds [here](https://github.com/mir-group/flare/actions).) Other C++ compilers may work, but we can't guarantee this.

### Operating systems
flare++ is tested on a Linux operating system (Ubuntu 20.04.3), but should also be compatible with Mac and Windows operating systems. If you run into issues running the code on Mac or Windows, please post to the [issue board](https://github.com/mir-group/flare/issues).

### Hardware requirements
There are no non-standard hardware requirements to download the software and train simple models&mdash;the introductory tutorial can be run on a single cpu. To train large models (10k+ sparse environments), we recommend using a compute node with at least 100GB of RAM.

## Tests
We recommend running unit tests to confirm that FLARE is running properly on your machine. We have implemented our tests using the pytest suite. You can call `pytest` from the command line in the tests directory.

Instructions (either DFT package will suffice):
```
pip install pytest
cd tests
pytest
```

## References
If you use FLARE++ including B2 descriptors, NormalizedDotProduct kernel and Sparse GP, please cite the following paper:

  > [1] Vandermause, J., Xie, Y., Lim, J.S., Owen, C.J. and Kozinsky, B., 2021. *Active learning of reactive Bayesian force fields: Application to heterogeneous hydrogen-platinum catalysis dynamics.* Nature Communications 13.1 (2022): 5183. https://www.nature.com/articles/s41467-022-32294-0

If you use FLARE active learning workflow, full Gaussian process or 2-body/3-body kernel in your research, please cite the following paper:

  > [2] Vandermause, J., Torrisi, S. B., Batzner, S., Xie, Y., Sun, L., Kolpak, A. M. & Kozinsky, B. *On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events.* npj Comput Mater 6, 20 (2020). https://doi.org/10.1038/s41524-020-0283-z

If you use FLARE LAMMPS pair style or MGP (mapped Gaussian process), please cite the following paper:

  > [3] Xie, Y., Vandermause, J., Sun, L. et al. *Bayesian force fields from active learning for simulation of inter-dimensional transformation of stanene.* npj Comput Mater 7, 40 (2021). https://doi.org/10.1038/s41524-021-00510-y

If you use FLARE PyLAMMPS for training, please cite the following paper:

  > [4] Xie, Y., Vandermause, J., Ramakers, S., Protik, N.H., Johansson, A. and Kozinsky, B., 2022. *Uncertainty-aware molecular dynamics from Bayesian active learning: Phase Transformations and Thermal Transport in SiC.* npj Comput. Mater. 9(1), 36 (2023).

If you use FLARE LAMMPS Kokkos pair style with GPU acceleration, please cite the following paper:

  > [5] Johansson, A., Xie, Y., Owen, C.J., Soo, J., Sun, L., Vandermause, J. and Kozinsky, B., 2022. *Micron-scale heterogeneous catalysis with Bayesian force fields from first principles and active learning.* arXiv preprint arXiv:2204.12573.
