import numpy as np
import sys
sys.path.append('../../../modules')
import crystals
import qe_input
import os

# calculation type
calculation = 'md'

# file locations
input_file_name = 'C.in'
output_file_name = 'C.out'
pw_loc = '/n/holylfs/LABS/kozinsky_lab/Software/qe-6.2.1/bin/pw.x'
pseudo_dir = '/n/holylfs/LABS/kozinsky_lab/Software/qe-6.2.1/pseudo'
outdir = './output'

# structure information
cube_lat = 2 * 1.763391008
cell = np.eye(3) * cube_lat
positions = crystals.cubic_diamond_positions(cube_lat)
nat = len(positions)
ntyp = 1
species = ['C'] * nat
ion_names = ['C']
ion_masses = [12]
ion_pseudo = ['C.pz-rrkjus.UPF']

# converged parameters
nk = 5
kvec = np.array([nk, nk, nk])
ecutwfc = 50
ecutrho = 4 * ecutwfc

scf_inputs = dict(pseudo_dir=pseudo_dir,
                  outdir=outdir,
                  nat=nat,
                  ntyp=ntyp,
                  ecutwfc=ecutwfc,
                  ecutrho=ecutrho,
                  cell=cell,
                  species=species,
                  positions=positions,
                  kvec=kvec,
                  ion_names=ion_names,
                  ion_masses=ion_masses,
                  ion_pseudo=ion_pseudo)

# set md details
dt = 20
nstep = 1000
ion_temperature = 'rescale-v'
tempw = 1000

md_inputs = dict(dt=dt,
                 nstep=nstep,
                 ion_temperature=ion_temperature,
                 tempw=tempw)

# make input file and run QE
calc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                        calculation, scf_inputs, md_inputs)
# calc.run_espresso()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')
