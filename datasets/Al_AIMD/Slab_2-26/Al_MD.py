import numpy as np
import sys
import os
import qe_input
import crystals
from ase.build import fcc111, add_adsorbate

# -----------------------------------------------------------------------------
#                         1. set positions and cell
# -----------------------------------------------------------------------------

# slab parameters
symbol = 'Al'
size = (3, 3, 4)
alat = 4.046
vacuum = 5
height = alat / 2  # vertical distance of adatom from last layer

# make slab with ase
slab = fcc111(symbol, size, a=alat, vacuum=vacuum)
add_adsorbate(slab, symbol, height, 'hcp')

positions = slab.positions
cell = slab.cell

# -----------------------------------------------------------------------------
#                         2. make qe input
# -----------------------------------------------------------------------------


input_file_name = 'Al_md.in'
output_file_name = 'Al_md.out'
outdir = './output'
pw_loc = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
pseudo_dir = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
calculation = 'md'


# setup qe file
nat = len(positions)
ntyp = 1
species = ['Al'] * nat
ion_names = ['Al']
mass = 27  # in amu
ion_masses = [mass]
ion_pseudo = ['Al.pbe-n-kjpaw_psl.1.0.0.UPF']
kvec = np.array([4, 4, 2])
ecutwfc = 29  # minimum recommended
ecutrho = 143  # minimum recommended

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
nstep = 100000
ion_temperature = 'initial'
tempw = 300

md_inputs = dict(dt=dt,
                 nstep=nstep,
                 ion_temperature=ion_temperature,
                 tempw=tempw)

# make input file
calc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                        calculation, scf_inputs, md_inputs,
                        metal=True)

# -----------------------------------------------------------------------------
#                           3. make bash file
# -----------------------------------------------------------------------------


# make bash file
bash_name = 'Al_md.sh'
command = 'mpirun {0} < {1} > {2}'.format(pw_loc, input_file_name,
                                          output_file_name)

bash_inputs = dict(n=32,
                   N=1,
                   t=14,
                   e='test.err',
                   p='kozinsky',
                   o='test.out',
                   mem_per_cpu=5000,
                   mail_user='jonathan_vandermause@g.harvard.edu',
                   command=command)

bash = qe_input.BashInput(bash_name, bash_inputs)
bash.write_bash_text()
