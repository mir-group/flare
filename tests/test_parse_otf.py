import os
import sys
import numpy as np

sys.path.append('../modules')
#from analyze_otf import parse_md_information, parse_dft_information, \
#    parse_header_information

from otf_parser import OtfAnalysis

sys.path.append('../otf_engine')

from kernels import two_plus_three_body, two_plus_three_body_grad
from env import AtomicEnvironment
"""
def test_parse_md_simple():
    _, _, _, _, _ = parse_md_information('h2_otf.out')


def test_parse_dft_simple():
    _, _, _, _ = parse_dft_information('h2_otf.out')

"""
def test_parse_header():

    os.system('cp test_files/sample_h2_otf.out .')

    header_dict = OtfAnalysis('sample_h2_otf.out').header

    assert header_dict['frames'] == 20
    assert header_dict['atoms'] == 2
    assert header_dict['species_set'] == {'H'}
    assert header_dict['dt'] == .0001
    assert header_dict['kernel'] == 'two_body'
    assert header_dict['n_hyps'] == 3
    assert header_dict['algo'] == 'L-BFGS-B'
    assert np.equal(header_dict['cell'],5*np.eye(3)).all()

    header_dict = OtfAnalysis('al_otf.out').header

    assert header_dict['frames'] == 100
    assert header_dict['atoms'] == 4
    assert header_dict['species_set'] == {'Al'}
    assert header_dict['dt'] == .001
    assert header_dict['kernel'] == 'three_body'
    assert header_dict['n_hyps'] == 3
    assert header_dict['algo'] == 'L-BFGS-B'
    assert np.equal(header_dict['cell'],3.9*np.eye(3)).all()


"""
def test_parse_dft():

    species, positions, forces, velocities = \
        parse_dft_information('h2_otf.out')

    assert species == [['H', 'H']]*11

    assert len(positions) == 11

    pos1 = np.array([np.array([2.3, 2.50000000, 2.50000000]),
                     np.array([2.8, 2.50000000, 2.50000000])])

    pos2 = np.array([np.array([2.29784856, 2.50000000, 2.50000000]),
                     np.array([2.80215144, 2.50000000, 2.50000000])])

    positions = np.array(positions)

    assert np.isclose(positions[0], pos1).all()
    assert np.isclose(positions[1], pos2).all()

    force1 = np.array([[-22.29815461, 0.00000000, 0.00000000],
                       [22.29815461, 0.00000000, 0.00000000]])

    forces = np.array(forces)

    assert np.isclose(forces[0], force1).all()
"""

def test_otf_gp_parser_h2_gp():
    """
    Test the capability of otf parser to read GP/DFT info
    :return:
    """
    os.system('cp test_files/sample_h2_otf.out .')
    parsed = OtfAnalysis('sample_h2_otf.out')
    assert (parsed.gp_species_list == [['H', 'H']] * 11)

    gp_positions = parsed.gp_position_list
    assert len(gp_positions) == 11


    pos1 = np.array([np.array([2.3, 2.50000000, 2.50000000]),
                     np.array([2.8, 2.50000000, 2.50000000])])

    pos2 = np.array([np.array([2.29784856, 2.50000000, 2.50000000]),
                     np.array([2.80215144, 2.50000000, 2.50000000])])

    assert np.isclose(gp_positions[0], pos1).all()
    assert np.isclose(gp_positions[1], pos2).all()

    force1 = np.array([[-22.29815461, 0.00000000, 0.00000000],
                       [22.29815461, 0.00000000, 0.00000000]])

    forces = parsed.gp_force_list




    assert np.isclose(forces[0], force1).all()


def test_otf_parser_h2_md():
    """
    Test the capability of otf parser to read MD info
    :return:
    """

    #TODO: Expand
    parsed = OtfAnalysis('sample_h2_otf.out')

    pos_frame_1 = np.array([np.array([2.29784856, 2.50000000, 2.50000000]),
                    np.array([2.80215144, 2.50000000, 2.50000000])])

    positions = parsed.position_list
    assert len(positions) == 19
    assert np.isclose(positions[0], pos_frame_1).all()


def test_output_md_structures():

    os.system('cp test_files/sample_h2_otf.out .')
    parsed = OtfAnalysis('sample_h2_otf.out')

    positions = parsed.position_list
    forces = parsed.force_list
    uncertainties = parsed.uncertainty_list
    lattice = parsed.header['cell']

    structures = parsed.output_md_structures()


    assert np.isclose(structures[-1].positions,positions[-1]).all()
    assert np.isclose(structures[-1].forces,forces[-1]).all()



def predict_on_structure(structure,gp):
    """
    Helper function for test_replicate_gp
    :param structure:
    :return:
    """
    forces = np.zeros(shape=(structure.nat,3))
    stds = np.zeros(shape=(structure.nat,3))
    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            forces[n][i] = float(force)
            stds[n][i] = np.sqrt(np.abs(var))

    return forces, stds


def test_replicate_gp():
    """
    Based on gp_test_al.out, ensures that given hyperparameters and DFT calls
    a GP model can be reproduced and correctly re-predict forces and
    uncertainties
    :return:
    """

    os.system('cp test_files/gp_test_al.out .')
    parsed = OtfAnalysis('gp_test_al.out')

    positions = parsed.position_list
    forces = parsed.force_list
    uncertainties = parsed.uncertainty_list
    lattice = parsed.header['cell']

    kernel = two_plus_three_body
    kernel_grad = two_plus_three_body_grad

    gp_model = parsed.make_gp(kernel=two_plus_three_body,
                              kernel_grad=two_plus_three_body_grad)

    assert len(gp_model.training_data) == 12

    structures = parsed.output_md_structures()

    assert np.isclose(structures[-1].positions, positions[-1]).all()
    assert np.isclose(structures[-1].forces, forces[-1]).all()

    final_structure = structures[-1]

    pred_for,pred_stds = predict_on_structure(final_structure,gp_model)

    assert np.isclose(final_structure.forces,pred_for).all()
    assert np.isclose(final_structure.stds,pred_stds).all()







