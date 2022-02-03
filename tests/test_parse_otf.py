import os
import sys
import numpy as np
from flare.io.otf_parser import OtfAnalysis
from flare.descriptors.env import AtomicEnvironment
from flare.bffs.gp.predict import predict_on_structure
from ase.io import read


def test_parse_header():

    os.system("cp test_files/sample_slab_otf.out .")

    header_dict = OtfAnalysis("sample_slab_otf.out").header

    assert header_dict["frames"] == 5000
    assert header_dict["atoms"] == 28
    assert header_dict["species_set"] == {"Al"}
    assert header_dict["dt"] == 0.001
    assert header_dict["kernel_name"] == "two_plus_three_body"
    assert header_dict["n_hyps"] == 5
    assert header_dict["algo"] == "BFGS"
    assert np.equal(
        header_dict["cell"],
        np.array([[8.59135, 0.0, 0.0], [4.29567, 7.44033, 0.0], [0.0, 0.0, 26.67654]]),
    ).all()

    os.system("rm sample_slab_otf.out")


def test_gp_parser():
    """
    Test the capability of otf parser to read GP/DFT info
    :return:
    """
    os.system("cp test_files/sample_slab_otf.out .")
    parsed = OtfAnalysis("sample_slab_otf.out")
    assert parsed.gp_species_list == [["Al"] * 28] * 4

    gp_positions = parsed.gp_position_list
    assert len(gp_positions) == 4

    pos1 = 1.50245891
    pos2 = 10.06179079
    assert pos1 == gp_positions[0][-1][1]
    assert pos2 == gp_positions[-1][0][2]

    force1 = 0.29430943
    force2 = -0.02350709
    assert force1 == parsed.gp_force_list[0][-1][1]
    assert force2 == parsed.gp_force_list[-1][0][2]

    os.system("rm sample_slab_otf.out")


def test_md_parser():
    """
    Test the capability of otf parser to read MD info
    :return:
    """
    os.system("cp test_files/sample_slab_otf.out .")
    parsed = OtfAnalysis("sample_slab_otf.out")

    pos1 = 10.09769665
    assert pos1 == parsed.position_list[0][0][2]
    assert len(parsed.position_list[0]) == 28

    os.system("rm sample_slab_otf.out")


def test_output_md_structures():

    os.system("cp test_files/sample_slab_otf.out .")
    parsed = OtfAnalysis("sample_slab_otf.out")

    positions = parsed.position_list
    forces = parsed.force_list

    structures = parsed.output_md_structures()

    assert np.isclose(structures[-1].positions, positions[-1]).all()
    assert np.isclose(structures[-1].forces, forces[-1]).all()

    os.system("rm sample_slab_otf.out")


def test_replicate_gp():
    """
    Based on gp_test_al.out, ensures that given hyperparameters and DFT calls
    a GP model can be reproduced and correctly re-predict forces and
    uncertainties
    :return:
    """

    os.system("cp test_files/sample_h2_otf.out .")
    parsed = OtfAnalysis("sample_h2_otf.out")

    positions = parsed.position_list
    forces = parsed.force_list

    gp_model = parsed.make_gp()

    structures = parsed.output_md_structures()

    assert np.isclose(structures[-1].positions, positions[-1]).all()
    assert np.isclose(structures[-1].forces, forces[-1]).all()

    final_structure = structures[-1]

    pred_for, pred_stds = predict_on_structure(final_structure, gp_model)

    assert np.isclose(final_structure.forces, pred_for).all()
    assert np.isclose(final_structure.stds, pred_stds).all()

    set_of_structures = structures[-3:-1]
    for structure in set_of_structures:
        pred_for, pred_stds = predict_on_structure(structure, gp_model)
        assert np.isclose(structure.forces, pred_for, atol=1e-6).all()
        assert np.isclose(structure.stds, pred_stds, atol=1e-6).all()
    os.system("rm sample_h2_otf.out")


def test_otf2xyz():
    xyz_file = "h2.xyz"
    parsed = OtfAnalysis("test_files/sample_h2_otf.out", calculate_energy=True)
    parsed.to_xyz(xyz_file)
    xyz_trj = read(xyz_file, index=":")
    assert np.allclose(xyz_trj[-1].positions[0, 0], 2.2794)
    assert np.allclose(xyz_trj[-2].get_forces()[-1, 2], 0.0000)
    os.system("rm h2.xyz")
