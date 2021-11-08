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

### Compilation for GPU with Kokkos
Run the above steps, follow [LAMMPS's compilation instructions with Kokkos](https://docs.lammps.org/Build_extras.html#kokkos).

If you really, really, really want to use the old Makefile system, it should work to copy the files from `kokkos/` into `/path/to/lammps/src`, do `make yes-kokkos` and otherwise follow the LAMMPS Kokkos instructions.

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
