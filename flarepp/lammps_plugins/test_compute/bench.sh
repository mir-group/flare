#!/bin/bash
set -e

for L in 1 2 5 10
do
    echo "L = $L, $(( 8 * $L * $L * $L )) atoms"
    echo "Serial"
    ~/lammps/build/lmp -sf kk -pk kokkos newton on neigh half -k on t 1 -var L $L -in in.bench | grep ns/day
    echo "MPI"
    mpirun ~/lammps/build/lmp -sf kk -pk kokkos newton on neigh half -k on t 1 -var L $L -in in.bench | grep ns/day
    echo "OpenMP"
    ~/lammps/ompbuild/lmp -sf kk -pk kokkos newton on neigh half -k on t 12 -var L $L -in in.bench | grep ns/day
    echo "CUDA"
    ~/lammps/cudabuild/lmp -sf kk -pk kokkos newton on neigh half -k on g 1 -var L $L -in in.bench | grep ns/day
    echo
done
