#!/bin/bash -l
#SBATCH -n 64
#SBATCH -N 4
#SBATCH -p batch
#SBATCH -o aimd_pert.out
#SBATCH -e aimd_pert.err
#SBATCH -J aimd_pert

module purge
module load espresso-6.1-gnu

mpirun -np 64 /home/homesoftware/espresso-6.1/qe-6.1/bin/pw.x -npool 2 -in Al_md_pert.in > Al_md_pert.out
