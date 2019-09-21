#!/bin/sh
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -t 10-00:00
#SBATCH -e test.err
#SBATCH -p kozinsky
#SBATCH -o test.out     
#SBATCH --job-name="test"
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xiey@g.harvard.edu

module load intel intel-mkl impi
module load Anaconda3/5.0.1-fasrc01
source activate venv

#srun -n 160 --mpi=pmi2 /n/home08/xiey/q-e/bin/pw.x -npool 10 -in scf.pwi > pwscf.out
#python relaunch.py
#python train_formate.py
python otf_setup.py
