#!/bin/sh
#SBATCH -n 64
#SBATCH -N 2
#SBATCH -t 24:00:00
#SBATCH -e mff_test.err
#SBATCH -p kozinsky
#SBATCH -o mff_test.out
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xiey@g.harvard.edu

module load gcc/4.9.3-fasrc01 openmpi/2.1.0-fasrc01
module load Anaconda3/5.0.1-fasrc02
source activate venv

python test.py Al2+3.npy 31
