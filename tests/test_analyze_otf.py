#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" OTF Parsing test suite based on py.test

Steven Torrisi
"""

import os
import sys
import numpy as np

sys.path.append('../modules')
from analyze_otf import parse_md_information, parse_dft_information, \
    parse_header_information


def test_parse_md_simple():
    os.system('cp test_files/otf_output_1.out otf_run.out')
    _, _, _, _ = parse_md_information('otf_run.out')
    os.system('rm otf_run.out')


def test_parse_dft_simple():
    os.system('cp test_files/otf_output_1.out otf_run.out')
    _, _, _, _ = parse_dft_information('otf_run.out')
    os.system('rm otf_run.out')


def test_parse_header_2():
    os.system('cp test_files/otf_output_2.out otf_run.out')

    header_dict = parse_header_information('otf_run.out')

    assert header_dict['frames'] == 20
    assert header_dict['atoms'] == 2
    assert header_dict['cutoff'] == 5
    assert header_dict['species'] == {'H'}
    assert header_dict['dt'] == .0001

    os.system('rm otf_run.out')


def test_parse_dft_2():
    os.system('cp test_files/otf_output_2.out otf_run.out')

    lattices, species, positions, forces = parse_dft_information('otf_run.out')

    assert np.equal(lattices, [5. * np.eye(3), 5. * np.eye(3)]).all()
    assert species == [['H', 'H'], ['H', 'H']]

    assert len(positions) == 2

    pos1 = np.array([np.array([2.51857000, 2.50000000, 2.50000000]),
                     np.array([4.48143000, 2.50000000, 2.50000000])])

    pos2 = np.array([np.array([2.55027962, 2.50000000, 2.50000000]),
                     np.array([4.44972038, 2.50000000, 2.50000000])])

    positions = np.array(positions)

    assert np.isclose(positions[0], pos1).all()
    assert np.isclose(positions[1], pos2).all()

    force1 = np.array([[1.90621314, 0.00000000, 0.00000000],
                       [-1.90621314, 0.00000000, 0.00000000]])


    forces = np.array(forces)

    assert np.isclose(forces[0], force1).all()
