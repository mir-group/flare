import numpy as np
from flare.io.parsers import parse_otf, parse_outfiles
from ase.io.vasp import read_vasp_out


def test_outfile_parser():
    output_dir = "test_files/output_files/"
    dft_data = parse_outfiles(output_dir, read_vasp_out)


def test_otf_parser():
    dft_data, gp_data = parse_otf("./test_files/output.out")

    assert len(gp_data["positions"]) == 12
    assert len(dft_data["positions"]) == 2
    assert dft_data["energies"][1] == -1137.336261
    assert dft_data["stresses"][1][2] == 9.655
    assert gp_data["store_hyps"][1][3] == -0.0038
