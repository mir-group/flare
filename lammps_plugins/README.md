# LAMMPS Plugin

## Compilation

1. Download LAMMPS, e.g.
```
git clone --depth=1 https://github.com/lammps/lammps.git
```
2. From this directory, run the install script.
```
./install.sh /path/to/lammps
```
This copies the necessary files and slightly updates the LAMMPS CMake file.

3. Compile LAMMPS with CMake as usual. If you're not using Kokkos, the Makefile system might also work.

*Note: The pair style uses the [Eigen](https://gitlab.com/libeigen/eigen) (header-only) library. This needs to be found by the compiler. Specifically, the `Eigen` folder. A simple workaround is usually to just clone the `eigen` repository, and `mv eigen/Eigen /path/to/lammps/src` if it is not easily available on your system.*

### Compilation for GPU with Kokkos
Run the above steps, follow [LAMMPS's compilation instructions with Kokkos](https://docs.lammps.org/Build_extras.html#kokkos). Typically
```
cd lammps
mkdir build
cd build
cmake ../cmake -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON [-DKokkos_ARCH_VOLTA70=on if not auto-detected]
```

If you really, really, really want to use the old Makefile system, you should be able to copy the files from `kokkos/` into `/path/to/lammps/src`, do `make yes-kokkos` and otherwise follow the LAMMPS Kokkos instructions.

## Basic usage
Input script:

```
newton on
pair_style	flare
pair_coeff	* * Si.txt
```

where `Si.txt` should be replaced by the name of your mapped model. Then run `lmp -in in.script` as usual.

### Running on a GPU with Kokkos
See the [LAMMPS documentation](https://docs.lammps.org/Speed_kokkos.html). In general, run
```
lmp -k on g 1 -sf kk -pk kokkos newton on neigh full -in in.script
```
When running with MPI, replace `g 1` with the number of GPUs *per node*.
