import numpy as np
import sys
import qe_parsers
from convergence import convergence


# parse 4000K file to retrieve perturbed positions
data_file = 'C.out'
steps = qe_parsers.parse_md_output(data_file)

# set up convergence run
txt_name = 'C_conv_output.txt'
input_file_string = './C_conv'
output_file_string = './C_conv'
pw_loc = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
pseudo_dir = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
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
nk = 5
kvec = np.array([nk, nk, nk])
ecutwfc = 100
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

# choose parameter grid
nks = [3, 4, 5]
ecutwfcs = [50, 75, 150]
rho_facs = [4, 8]

convergence(txt_name, input_file_string, output_file_string, pw_loc,
            calculation, scf_inputs, nks, ecutwfcs, rho_facs)
