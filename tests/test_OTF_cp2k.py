import pytest
import os
import sys
import numpy as np
from flare.otf import OTF
from flare.gp import GaussianProcess
from flare.struc import Structure
import flare.kernels as en

def cleanup(target: str = None):
    try:
        os.remove('cp2k.in')
        os.remove('cp2k-RESTART.wfn')
        if (target is not None):
            os.remove(target)
    except:
        pass

# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------
@pytest.mark.skipif(not os.environ.get('CP2K_COMMAND',
                          False), reason='CP2K_COMMAND not found '
                                  'in environment: Please install CP2K '
                                  'and set the CP2K_COMMAND env. '
                                  'variable to point to cp2k.popt')
def test_otf_h2():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    os.system('cp ./test_files/cp2k_input_1.in ./cp2k.in')

    cp2k_input = './cp2k.in'
    dt = 0.0001
    number_of_steps = 5
    cutoffs = np.array([5])
    dft_loc = os.environ.get('CP2K_COMMAND')
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

    otf = OTF(cp2k_input, dt, number_of_steps, gp, dft_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, max_atoms_added=1,
              dft_softwarename="cp2k",
              output_name='h2_otf_cp2k')

    otf.run()
    os.system('mkdir test_outputs')
    os.system('mv h2_otf_cp2k.* test_outputs')
    cleanup()

@pytest.mark.skipif(not os.environ.get('CP2K_COMMAND',
                          False), reason='CP2K_COMMAND not found '
                                  'in environment: Please install CP2K '
                                  ' and set the CP2K_COMMAND env. '
                                  'variable to point to pw.x.')
def test_otf_al():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    os.system('cp ./test_files/cp2k_input_2.in ./cp2k.in')

    cp2k_input = './cp2k.in'
    dt = 0.001
    number_of_steps = 5
    cutoffs = np.array([3.9, 3.9])
    dft_loc = os.environ.get('CP2K_COMMAND')
    std_tolerance_factor = 1
    max_atoms_added = 2
    freeze_hyps = 3

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

    otf = OTF(cp2k_input, dt, number_of_steps, gp, dft_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, output_name='al_otf_cp2k',
              freeze_hyps=freeze_hyps, skip=5,
              dft_softwarename="cp2k",
              max_atoms_added=max_atoms_added)

    otf.run()
    os.system('mkdir test_outputs')
    os.system('mv al_otf_cp2k.* test_outputs')

    cleanup()
