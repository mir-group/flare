import pytest
import os
import sys
import numpy as np
sys.path.append('../otf_engine')
from otf import OTF
from gp import GaussianProcess
from struc import Structure
import kernels as kern
from test_qe_util import cleanup_espresso_run


# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------

def test_otf_1_1():
    """
    Test that a minimal OTF run can complete after two steps
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    qe_input = './pwscf.in'
    dt = 0.0001
    number_of_steps = 2
    cutoff = 4
    pw_loc = os.environ.get('PWSCF_COMMAND')
    std_tolerance_factor = -0.1

    # make gp model
    kernel_name = 'n_body_sc'
    kernel = kern.n_body_sc
    kernel_grad = kern.n_body_sc_grad
    hyps = np.array([1, 1, 1])
    hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
    energy_force_kernel = kern.energy_force_sc

    gp = \
        GaussianProcess(kernel_name=kernel_name, kernel=kernel,
                        kernel_grad=kernel_grad,
                        hyps=hyps,
                        hyp_labels=hyp_labels,
                        energy_force_kernel=energy_force_kernel)

    otf = OTF(qe_input, dt, number_of_steps, gp, cutoff, pw_loc,
              std_tolerance_factor)
    otf.run()


def test_otf_1_2():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in ./pwscf.in')

    qe_input = './pwscf.in'
    dt = 0.0001
    number_of_steps = 20
    cutoff = 5
    pw_loc = os.environ.get('PWSCF_COMMAND')
    std_tolerance_factor = -0.1

    # make gp model
    kernel_name = 'n_body_sc'
    kernel = kern.n_body_sc
    kernel_grad = kern.n_body_sc_grad
    hyps = np.array([1, 1, 1])
    hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
    energy_force_kernel = kern.energy_force_sc

    gp = \
        GaussianProcess(kernel_name=kernel_name, kernel=kernel,
                        kernel_grad=kernel_grad,
                        hyps=hyps,
                        hyp_labels=hyp_labels,
                        energy_force_kernel=energy_force_kernel)

    otf = OTF(qe_input, dt, number_of_steps, gp, cutoff, pw_loc,
              std_tolerance_factor)

    otf.run()
