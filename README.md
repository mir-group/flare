[![Build Status](https://github.com/mir-group/flare_pp/actions/workflows/main.yml/badge.svg)](https://github.com/mir-group/flare_pp/actions)

# flare++
Major features:
* Bayesian force fields based on sparse Gaussian process regression.
* Multielement many-body descriptors based on the atomic cluster expansion.
* Mapping to efficient parametric models.
* Coupling to LAMMPS for large-scale molecular dynamics simulations.

Check out our preprint introducing flare++ [here](https://arxiv.org/abs/2106.01949).

## Demo and Instructions for Use
An introductory tutorial in Google Colab is available [here](https://colab.research.google.com/drive/18_pTcWM19AUiksaRyCgg9BCpVyw744xv). The tutorial shows how to run flare++ on energy and force data, demoing "offline" training on the MD17 dataset and "online" on-the-fly training of a simple aluminum force field. A video walkthrough of the tutorial, including detailed discussion of expected outputs, is available [here](https://www.youtube.com/watch?v=-FH_VqRQrso&t=875s).

The tutorial takes a few minutes to run on a normal desktop computer or laptop (excluding installation time).

## Installation guide
The easiest way to install is with pip:
```
pip install flare_pp
```
This will take a few minutes on a normal desktop computer or laptop.

If you're installing on Harvard's compute cluster, make sure to load the following modules first:
```
module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01
```

### Compiling LAMMPS
See [lammps_plugins/README.md](https://github.com/mir-group/flare_pp/blob/master/lammps_plugins/README.md).

## System requirements
### Software dependencies
* cmake>=3.14.5 (to compile the C++ source code)
* [flare](https://github.com/mir-group/flare) (for on-the-fly training)

### Operating systems
flare++ is tested with Github Actions on a Linux operating system (Ubuntu 20.04.3). You can find a summary of recent builds [here](https://github.com/mir-group/flare_pp/actions).

We expect flare++ to be compatible with Mac and Windows operating systems, but can't guarantee this. If you run into issues running the code on Mac or Windows, please post to the issue board.

### Hardware requirements
There are no non-standard hardware requirements to download the software and train simple models&mdash;the introductory tutorial can be run on a single cpu. To train large models (10k+ sparse environments), we recommend using a compute node with at least 100GB of RAM.

## Documentation
Preliminary documentation of the C++ source code can be accessed [here](https://mir-group.github.io/flare_pp/). 
