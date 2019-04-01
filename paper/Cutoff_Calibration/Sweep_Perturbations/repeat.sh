#!/bin/sh
#SBATCH -n 128
#SBATCH -N 4
#SBATCH -t 14-00:00
#SBATCH -e test.err
#SBATCH -p kozinsky
#SBATCH -o test.out
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonathan_vandermause@g.harvard.edu

module load gcc/4.9.3-fasrc01 openmpi/2.1.0-fasrc01
module load python/3.6.3-fasrc01


source activate numba
python sweep_scf.py