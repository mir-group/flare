#!/bin/bash

set -e -x

if [ "$#" -ne 1 ]; then
    echo "Give the path to lammps as a command-line argument!"
    exit 1
fi

lammps=$1

cd $lammps
git stash
git clean -df
