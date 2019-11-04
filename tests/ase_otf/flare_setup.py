import numpy as np

from flare import otf, kernels
from flare.gp import GaussianProcess
from flare.mgp.mgp import MappedGaussianProcess
from flare.ase.calculator import FLARE_Calculator
import flare.mc_simple as mc_simple

# ---------- create gaussian process model -------------------
kernel = mc_simple.two_plus_three_body_mc
kernel_grad = mc_simple.two_plus_three_body_mc_grad
energy_force_kernel = mc_simple.two_plus_three_mc_force_en
energy_kernel = mc_simple.two_plus_three_mc_en

hyps = np.array([0.1, 1., 0.001, 1, 0.03])
two_cut = 4.0
three_cut = 4.0
cutoffs = np.array([two_cut, three_cut])
hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
opt_algorithm = 'BFGS'

gp_model = GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                           energy_kernel=energy_kernel,
                           energy_force_kernel=energy_force_kernel,
                           hyp_labels=hyp_labels,
                           opt_algorithm=opt_algorithm, par=True)

# ----------- create mapped gaussian process ------------------
struc_params = {'species': [6],
                'cube_lat': np.eye(3) * 100,
                'mass_dict': {'0': 12.0107}}

# grid parameters
lower_cut = 1.0
grid_num_2 = 64
grid_num_3 = 64
grid_params = {'bounds_2': [[lower_cut], [two_cut]],
               'bounds_3': [[lower_cut, lower_cut, 0],
                            [three_cut, three_cut, np.pi]],
               'grid_num_2': grid_num_2,
               'grid_num_3': [grid_num_3, grid_num_3, grid_num_3],
               'svd_rank_2': 0,
               'svd_rank_3': 0,
               'bodies': [2, 3],
               'load_grid': None,
               'update': True}

mgp_model = MappedGaussianProcess(gp_model.hyps, gp_model.cutoffs,
            grid_params, struc_params, mean_only=False, container_only=False,
            GP=gp_model, lmp_file_name='lmp.mgp')

# ------------ create ASE's flare calculator -----------------------
flare_calc = FLARE_Calculator(gp_model, mgp_model, par=True, use_mapping=True)
