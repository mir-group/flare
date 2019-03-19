import pytest
import os
import sys
import numpy as np
sys.path.append('../otf_engine')
from otf import OTF
from gp import GaussianProcess
from struc import Structure
import kernels as en
from test_qe_util import cleanup_espresso_run


# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------

def test_otf_h2():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    qe_input = './pwscf.in'
    dt = 0.0001
    number_of_steps = 20
    cutoffs = np.array([5])
    pw_loc = os.environ.get('PWSCF_COMMAND')
    std_tolerance_factor = -0.1

    # make gp model
    kernel = en.two_body
    kernel_grad = en.two_body_grad
    hyps = np.array([1, 1, 1])
    hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
    energy_force_kernel = en.two_body_force_en

    gp = \
        GaussianProcess(kernel=kernel,
                        kernel_grad=kernel_grad,
                        hyps=hyps,
                        cutoffs=cutoffs,
                        hyp_labels=hyp_labels,
                        energy_force_kernel=energy_force_kernel,
                        maxiter=50)

    otf = OTF(qe_input, dt, number_of_steps, gp, pw_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, max_atoms_added=1,
              output_name='h2_otf.out')

    otf.run()


def test_otf_al():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    os.system('cp ./test_files/qe_input_2.in ./pwscf.in')

    qe_input = './pwscf.in'
    dt = 0.001
    number_of_steps = 100
    cutoffs = np.array([3.9, 3.9])
    pw_loc = os.environ.get('PWSCF_COMMAND')
    std_tolerance_factor = -0.01

    # make gp model
    kernel = en.three_body
    kernel_grad = en.three_body_grad
    hyps = np.array([0.1, 1, 0.01])
    hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
    energy_force_kernel = en.three_body_force_en

    gp = \
        GaussianProcess(kernel=kernel,
                        kernel_grad=kernel_grad,
                        hyps=hyps,
                        cutoffs=cutoffs,
                        hyp_labels=hyp_labels,
                        energy_force_kernel=energy_force_kernel,
                        maxiter=50)

    otf = OTF(qe_input, dt, number_of_steps, gp, pw_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, output_name='al_otf.out',
              freeze_hyps=True)

    otf.run()
