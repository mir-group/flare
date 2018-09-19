#!/bin/bash -l
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -o otf_pert.out
#SBATCH -e otf_pert.err 
#SBATCH -J otf_pert

module purge
module load espresso-6.1-gnu

source activate py3.6
export PWSCF_COMMAND='mpirun -np 16 /home/homesoftware/espresso-6.1/qe-6.1/bin/pw.x'

python al_vac.py
exit
