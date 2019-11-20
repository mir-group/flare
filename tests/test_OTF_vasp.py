import pytest
import os
import sys
import numpy as np
from flare.otf import OTF
from flare.gp import GaussianProcess
from flare.struc import Structure
import flare.kernels as en

from flare.dft_interface.vasp_util import *


# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------

def test_otf_h2():
    """
    :return:
    """
    os.system('cp ./test_files/test_POSCAR_2 POSCAR')

    vasp_input = './POSCAR'
    dt = 0.0001
    number_of_steps = 5
    cutoffs = np.array([5])
    dft_loc = 'cp ./test_files/test_vasprun_h2.xml vasprun.xml'
    std_tolerance_factor = -0.1

    # make gp model
    kernel = en.two_body
    kernel_grad = en.two_body_grad
    hyps = np.array([1, 1, 1])
    hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
    energy_force_kernel = en.two_body_force_en

    gp = GaussianProcess(kernel=kernel,
                        kernel_grad=kernel_grad,
                        hyps=hyps,
                        cutoffs=cutoffs,
                        hyp_labels=hyp_labels,
                        energy_force_kernel=energy_force_kernel,
                        maxiter=50)

    otf = OTF(vasp_input, dt, number_of_steps, gp, dft_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, max_atoms_added=1,
              no_cpus=1, dft_softwarename='vasp',
              output_name='h2_otf_vasp')

    otf.run()

    os.system('mv h2_otf_vasp* test_outputs')



