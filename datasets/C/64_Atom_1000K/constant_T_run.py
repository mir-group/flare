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
sc_size = 2
sc_cell = sc_size * cell
sc_positions = crystals.get_supercell_positions(sc_size, cell, positions)
nat = len(sc_positions)
ntyp = 1
species = ['C'] * nat
ion_names = ['C']
ion_masses = [12]
ion_pseudo = ['C.pz-rrkjus.UPF']

# converged parameters
nk = 3
kvec = np.array([nk, nk, nk])
ecutwfc = 50
ecutrho = 4 * ecutwfc

scf_inputs = dict(pseudo_dir=pseudo_dir,
                  outdir=outdir,
                  nat=nat,
                  ntyp=ntyp,
                  ecutwfc=ecutwfc,
                  ecutrho=ecutrho,
                  cell=sc_cell,
                  species=species,
                  positions=sc_positions,
                  kvec=kvec,
                  ion_names=ion_names,
                  ion_masses=ion_masses,
                  ion_pseudo=ion_pseudo)

# set md details
dt = 20
nstep = 1000
ion_temperature = 'rescale-v'
tempw = 4000

md_inputs = dict(dt=dt,
                 nstep=nstep,
                 ion_temperature=ion_temperature,
                 tempw=tempw)

# make input file
calc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                        calculation, scf_inputs, md_inputs)

# make bash file
bash_name = 'constant_T_run.sh'
npool = 10
command = 'mpirun {0} -npool {1} < {2} > {3}'.format(pw_loc, npool,
                                                     input_file_name,
                                                     output_file_name)

bash_inputs = dict(n=180,
                   N=6,
                   t=7,
                   e='test.err',
                   p='kozinsky',
                   o='test.out',
                   mem_per_cpu=1000,
                   mail_user='jonathan_vandermause@g.harvard.edu',
                   command=command)

bash = qe_input.BashInput(bash_name, bash_inputs)
bash.write_bash_text()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')
