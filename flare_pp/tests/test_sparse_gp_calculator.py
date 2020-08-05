import numpy as np
from flare_pp.sparse_gp import SparseGP
from flare_pp.sparse_gp_calculator import SGP_Calculator
from build._C_flare import DotProductKernel, B2_Calculator, SparseGP_DTC
from flare import struc
from test_sparse_gp import sgp_py, sgp_cpp

