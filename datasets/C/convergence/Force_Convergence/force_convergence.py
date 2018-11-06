import numpy as np
import sys
import qe_parsers
from convergence import convergence


# parse 4000K file to retrieve perturbed positions
data_file = 'C.out'
steps = qe_parsers.parse_md_output(data_file)

# set up convergence run
input_file_name = './C_conv.in'
output_file_name = './C_conv.out'
pw_loc = '/n/holylfs/LABS/kozinsky_lab/Software/qe-6.2.1/bin/pw.x'
pseudo_dir = '/n/holylfs/LABS/kozinsky_lab/Software/qe-6.2.1/pseudo'
outdir = './output'
calculation = 'scf'

cube_lat = 2 * 1.763391008
cell = np.eye(3) * cube_lat
sc_size = 2
sc_cell = sc_size * cell
sc_positions = steps[500]['positions']

nat = len(sc_positions)
ntyp = 1
species = ['C'] * nat
ion_names = ['C']
ion_masses = [12]
ion_pseudo = ['C.pz-rrkjus.UPF']

# converged parameters
nk = 4
kvec = np.array([nk, nk, nk])
ecutwfc = 50
ecutrho = 8 * ecutwfc

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

# choose parameter grid
nks = [3]
ecutwfcs = [50]
rho_facs = [4]

convergence(input_file_name, output_file_name, pw_loc,
            calculation, scf_inputs, nks, ecutwfcs, rho_facs)
