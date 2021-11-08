[![Build Status](https://github.com/mir-group/flare_pp/actions/workflows/main.yml/badge.svg)](https://github.com/mir-group/flare_pp/actions)

# flare++
Documentation can be accessed [here](https://mir-group.github.io/flare_pp/). An introductory tutorial in Google Colab is available [here](https://colab.research.google.com/drive/18_pTcWM19AUiksaRyCgg9BCpVyw744xv).

## Installation

```
pip install flare_pp
```

If you're installing on Harvard's compute cluster, load the following modules first:
```
module load cmake/3.17.3-fasrc01 python/3.6.3-fasrc01 gcc/9.3.0-fasrc01
```

## Compiling LAMMPS

To compile lammps with the flare++ pair style, run the following sequence of commands:

```
cp lammps_plugins/{pair_flare*,lammps_descriptor*} lammps/src
cp src/flare_pp/{y_grad*,radial*,cutoffs*} lammps/src
sudo cp -r ${BUILD_DIR}/External/Eigen3/Eigen /usr/include
cd lammps/src
make mpi CCFLAGS='-std=c++11'
```
