#!/bin/bash

set -e -x

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
    ${LAMMPS_SOURCE_DIR}/cutoffs.h
    ${LAMMPS_SOURCE_DIR}/lammps_descriptor.cpp
    ${LAMMPS_SOURCE_DIR}/lammps_descriptor.h
    ${LAMMPS_SOURCE_DIR}/radial.cpp
    ${LAMMPS_SOURCE_DIR}/radial.h
    ${LAMMPS_SOURCE_DIR}/y_grad.cpp
    ${LAMMPS_SOURCE_DIR}/y_grad.h
)
if(PKG_KOKKOS)
    target_sources(lammps PRIVATE
        ${LAMMPS_SOURCE_DIR}/KOKKOS/cutoffs_kokkos.cpp
        ${LAMMPS_SOURCE_DIR}/KOKKOS/cutoffs_kokkos.h
        ${LAMMPS_SOURCE_DIR}/KOKKOS/lammps_descriptor_kokkos.cpp
        ${LAMMPS_SOURCE_DIR}/KOKKOS/lammps_descriptor_kokkos.h
        ${LAMMPS_SOURCE_DIR}/KOKKOS/radial_kokkos.cpp
        ${LAMMPS_SOURCE_DIR}/KOKKOS/radial_kokkos.h
        ${LAMMPS_SOURCE_DIR}/KOKKOS/y_grad_kokkos.cpp
        ${LAMMPS_SOURCE_DIR}/KOKKOS/y_grad_kokkos.h
    )
endif()
' >> $lammps/cmake/CMakeLists.txt
