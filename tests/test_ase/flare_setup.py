import sys
import numpy as np

sys.path.append('../../..')
from flare import gp, otf, kernels
from flare.ase.calculator import FLARE_Calculator
import flare.mc_simple as mc_simple

# create gaussian process model
kernel = mc_simple.two_plus_three_body_mc
kernel_grad = mc_simple.two_plus_three_body_mc_grad
hyps = np.array([0.1, 1., 0.001, 1, 0.06])
cutoffs = np.array([5., 5.])
hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
opt_algorithm = 'BFGS'

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                              hyp_labels=hyp_labels,
                              opt_algorithm=opt_algorithm, par=False)

# create mapped force field
# ...

# create ASE's flare calculator
flare_calc = FLARE_Calculator(gp_model, mgp_model=None, par=True, use_mapping=False)
