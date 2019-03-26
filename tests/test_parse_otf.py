import os
import sys
import numpy as np

sys.path.append('../modules')
from analyze_otf import parse_md_information, parse_dft_information, \
    parse_header_information

from otf_parser import OtfAnalysis


def test_parse_md_simple():
    _, _, _, _, _ = parse_md_information('h2_otf.out')


def test_parse_dft_simple():
    _, _, _, _ = parse_dft_information('h2_otf.out')


def test_parse_header():

    header_dict = parse_header_information('h2_otf.out')

    assert header_dict['frames'] == 20
    assert header_dict['atoms'] == 2
    assert header_dict['species'] == {'H'}
    assert header_dict['dt'] == .0001
    assert header_dict['kernel'] == 'two_body'
    assert header_dict['n_hyps'] == 3
    assert header_dict['algo'] == 'L-BFGS-B'

    header_dict = parse_header_information('al_otf.out')

    assert header_dict['frames'] == 100
    assert header_dict['atoms'] == 4
    assert header_dict['species'] == {'Al'}
    assert header_dict['dt'] == .001
    assert header_dict['kernel'] == 'three_body'
    assert header_dict['n_hyps'] == 3
    assert header_dict['algo'] == 'L-BFGS-B'




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


def test_otf_gp_parser_h2():
    """
    Short-term unit test to ensure that things are parsing correctly
    :return:
    """

    parsed = OtfAnalysis('h2_otf.out')
    assert(parsed.gp_species_list == [['H','H']]*11)

    positions = parsed.gp_position_list
    assert len(positions) == 11

    pos1 = np.array([np.array([2.3, 2.50000000, 2.50000000]),
                     np.array([2.8, 2.50000000, 2.50000000])])

    pos2 = np.array([np.array([2.29784856, 2.50000000, 2.50000000]),
                     np.array([2.80215144, 2.50000000, 2.50000000])])

    assert np.isclose(positions[0], pos1).all()
    assert np.isclose(positions[1], pos2).all()

    force1 = np.array([[-22.29815461, 0.00000000, 0.00000000],
                       [22.29815461, 0.00000000, 0.00000000]])

    forces = parsed.gp_force_list

    assert np.isclose(forces[0], force1).all()





