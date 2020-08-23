import pytest
import os
import numpy as np
from flare.otf import OTF
from flare.gp import GaussianProcess
from flare.struc import Structure

from flare.dft_interface.vasp_util import *


# ------------------------------------------------------
#                   test  otf runs
# ------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VASP_COMMAND", False),
    reason="VASP_COMMAND not found "
    "in environment: Please install VASP "
    " and set the VASP_COMMAND env. "
    "variable to point to cp2k.popt",
)
def test_otf_h2():
    """
    :return:
    """
    os.system("cp ./test_files/test_POSCAR_2 POSCAR")

    vasp_input = "./POSCAR"
    dt = 0.0001
    number_of_steps = 5
    cutoffs = {"twobody": 5}
    dft_loc = "cp ./test_files/test_vasprun_h2.xml vasprun.xml"
    std_tolerance_factor = -0.1

    # make gp model
    hyps = np.array([1, 1, 1])
    hyp_labels = ["Signal Std", "Length Scale", "Noise Std"]

    gp = GaussianProcess(
        kernel_name="2", hyps=hyps, cutoffs=cutoffs, hyp_labels=hyp_labels, maxiter=50
    )

    otf = OTF(
        dt=dt,
        number_of_steps=number_of_steps,
        gp=gp,
        calculate_energy=True,
        std_tolerance_factor=std_tolerance_factor,
        init_atoms=[0],
        output_name="h2_otf_vasp",
        max_atoms_added=1,
        force_source="vasp",
        dft_input=vasp_input,
        dft_loc=dft_loc,
        dft_output="vasprun.xml",
        n_cpus=1,
    )

    otf.run()

    os.system("mv h2_otf_vasp* test_outputs")
