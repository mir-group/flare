#!/bin/bash -l
#SBATCH -n 64
#SBATCH -N 4
#SBATCH -p batch
#SBATCH -o relax_pbe.out
#SBATCH -e relax_pbe.err 
#SBATCH -J relax_pbe

module purge
module load espresso-6.1-gnu

export PWSCF_COMMAND='mpirun -np 64 /home/homesoftware/espresso-6.1/qe-6.1/bin/pw.x'

$PWSCF_COMMAND -in Al_scf_relax.in > Al_scf_pbe_relax.out
exit
