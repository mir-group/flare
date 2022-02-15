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
