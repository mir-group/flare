import numpy as np
from flare_pp.parsing.simple_parser import simple_parser


def test_parse():
    dft_data, gp_data = simple_parser("test_files/output.out")

    assert len(gp_data["positions"]) == 12
    assert len(dft_data["positions"]) == 2
    assert dft_data["energies"][1] == -1137.336261
    assert dft_data["stresses"][1][2] == 9.655
    assert gp_data["store_hyps"][1][3] == -0.0038
