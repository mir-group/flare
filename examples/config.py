import numpy as np
from ase import units

# Super cell
ase_atoms_file = "POSCAR"
ase_atoms_format = "vasp"
replicate = [1, 1, 1]
jitter = 0.1

# MD
md_engine = "VelocityVerlet"
md_kwargs = {}
temperature = 1000
dt = 0.001 # timestep is 1 fs
number_of_steps = 5

# GP
kernels = ["twobody", "threebody"]
gp_parameters = {"cutoff_twobody": 10.0, "cutoff_threebody": 6.0}
random_init_hyps = True
components = "mc"  # multi-component. For single-comp, use 'sc'
opt_algorithm = "L-BFGS-B"
n_cpus = 1
par = False

# Mapped GP
use_mapping = False
grid_params = {
    "twobody": {"grid_num": [64]},
    "threebody": {"grid_num": [16, 16, 16]},
}
var_map = "pca"

# On-the-fly
output_name = "myotf"
init_atoms = [0, 1, 2, 3]
std_tolerance_factor = 1.0
max_atoms_added = -1
freeze_hyps = 10
write_model = 4

# DFT calculaotr
ase_dft_calc_name = "LennardJones"
ase_dft_calc_kwargs = {}
ase_dft_calc_params = {"sigma": 3.0, "rc": 9.0}
