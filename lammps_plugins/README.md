# LAMMPS Plugin

## Compilation

1. Download LAMMPS, e.g.
```
git clone --depth=1 https://github.com/lammps/lammps.git
```
2. From this directory (`flare/lammps_plugins`), run the install script.
```
./install.sh /path/to/lammps
```
This copies the necessary files and slightly updates the LAMMPS CMake file.

3. Compile LAMMPS with CMake as usual. If you're not using Kokkos, the Makefile system might also work.

*Note: The pair style uses the [Eigen](https://gitlab.com/libeigen/eigen) (header-only) library. This needs to be found by the compiler. Specifically, the `Eigen` folder. A simple workaround is usually to just clone the `eigen` repository, and `mv eigen/Eigen /path/to/lammps/src/` if it is not easily available on your system.*

### Compilation for GPU with Kokkos
Run the above steps, follow [LAMMPS's compilation instructions with Kokkos](https://docs.lammps.org/Build_extras.html#kokkos). Typically
```
cd lammps
mkdir build
cd build
cmake ../cmake -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON [-DKokkos_ARCH_VOLTA70=on if not auto-detected]
```

*Note: FLARE relies on [KokkosKernels](https://github.com/kokkos/kokkos-kernels) for a performance-portable matrix-matrix product. This will take advantage of MKL/cuBLAS etc. if found. By default, CMake will download KokkosKernels from GitHub and compile it together with LAMMPS. If you need multiple LAMMPS installations with the same Kokkos configuration, you can install Kokkos and KokkosKernels manually and then use the `-DEXTERNAL_KOKKOS=ON` option (with `-DCMAKE_PREFIX_PATH=/path/to/install` as needed).*

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

In order to run large systems, the atoms will be divided into batches to reduce memory usage. The batch size is controlled by the `MAXMEM` environment variable (in GB). If necessary, set this to an estimate of how much memory FLARE++ can use (i.e., total GPU memory minus LAMMPS's memory for neighbor lists etc.). If you are memory-limited, you can set `MAXMEM=1` or similar, otherwise leave it to a larger number for more parallelism. The default is 12 GB, which should work for most systems while not affecting performance.

`MAXMEM` is printed at the beginning of the simulation *from every MPI process*, in order to verify that the environment variable has been correctly set *on all nodes*. Look at `mpirun -x` if this is not the case.
