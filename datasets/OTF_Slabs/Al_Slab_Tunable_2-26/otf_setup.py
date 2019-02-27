import numpy as np
import sys
from ase.build import fcc111, add_adsorbate
import otf
import kernels
import gp
import qe_input
import initialize_velocities

# -----------------------------------------------------------------------------
#                         1. set initial positions
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
#            2. set initial velocities and previous positions
# -----------------------------------------------------------------------------

nat = len(positions)
temperature = 300  # in kelvin
mass = 27  # in amu

velocities = \
    initialize_velocities.get_random_velocities(nat, temperature, mass)

dt = 0.001  # in ps
previous_positions = positions - velocities * dt


# -----------------------------------------------------------------------------
#                     3. create gaussian process model
# -----------------------------------------------------------------------------

kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.01, 1, 0.001,
                 1, 0.05])
cutoffs = np.array([7, 4.5])
hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
energy_force_kernel = kernels.two_plus_three_force_en
energy_kernel = kernels.two_plus_three_en
opt_algorithm = 'BFGS'

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs, hyp_labels,
                              energy_force_kernel, energy_kernel,
                              opt_algorithm)


# -----------------------------------------------------------------------------
#                 4. make qe input file for initial structure
# -----------------------------------------------------------------------------

# setup qe file
calculation = 'scf'
input_file_name = 'slab.in'
output_file_name = 'pwscf.out'
pw_loc = '/n/home03/jonpvandermause/qe-6.2.1/bin/pw.x'
pseudo_dir = '/n/home03/jonpvandermause/qe-6.2.1/pseudo'
outdir = './output'
nat = len(positions)
ntyp = 1
species = ['Al'] * nat
ion_names = ['Al']
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

# make input file
calc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                        calculation, scf_inputs, metal=True)


# -----------------------------------------------------------------------------
#                            4. create otf object
# -----------------------------------------------------------------------------

number_of_steps = 100000
std_tolerance_factor = 1
init_atoms = [0]

otf_model = otf.OTF(input_file_name, dt, number_of_steps, gp_model,
                    pw_loc, std_tolerance_factor,
                    prev_pos_init=previous_positions, par=True, parsimony=True,
                    skip=1, init_atoms=init_atoms)
