import pytest
from subprocess import call
import os
import sys
import numpy as np
from flare.otf import OTF
from flare.gp import GaussianProcess
from flare.struc import Structure
import flare.kernels as en

def cleanup(target: list = None):
    os.remove('cp2k.in')
    os.remove('cp2k-RESTART.wfn')
    if (target is not None):
        for i in target:
            os.remove(i)

# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------
@pytest.mark.skipif(not os.environ.get('CP2K_COMMAND',
                          False), reason='CP2K_COMMAND not found '
                                  'in environment: Please install CP2K '
                                  'and set the CP2K_COMMAND env. '
                                  'variable to point to cp2k.popt')
def test_otf_h2_par():
    """
    Test that an otf run can survive going for more steps
    :return:
    """
    call('cp ./test_files/cp2k_input_1.in ./cp2k.in'.split())

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
                        par=True,
                        maxiter=50)

    otf = OTF(cp2k_input, dt, number_of_steps, gp, dft_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, max_atoms_added=1,
              dft_softwarename="cp2k",
              no_cpus=2,
              par=True, mpi="mpi",
              output_name='h2_otf_cp2k_par')

    otf.run()
    call('mkdir test_outputs'.split())
    call('mv h2_otf_cp2k_par* test_outputs'.split())
    cleanup()
