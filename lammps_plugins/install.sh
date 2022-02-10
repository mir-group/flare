#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Give the path to lammps as a command-line argument!"
    exit 1
fi

lammps=$1
src=$lammps/src
kk=$src/KOKKOS

for f in $(ls kokkos)
do
    ln -s $(pwd)/kokkos/$f $kk/$f
done

for f in *.cpp *.h
do
    ln -s $(pwd)/$f $src/$f
done

for f in cutoffs radial y_grad
do
    for ex in cpp h
    do
        ln -s $(pwd)/../src/flare_pp/$f.$ex $src/$f.$ex
    done
done

echo '
target_sources(lammps PRIVATE
    ${LAMMPS_SOURCE_DIR}/cutoffs.cpp
    ${LAMMPS_SOURCE_DIR}/lammps_descriptor.cpp
    ${LAMMPS_SOURCE_DIR}/radial.cpp
    ${LAMMPS_SOURCE_DIR}/y_grad.cpp
)
' >> $lammps/cmake/CMakeLists.txt
