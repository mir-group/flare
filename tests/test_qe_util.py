import pytest
import os
import sys
import numpy as np
sys.path.append('../otf_engine')
from struc import Structure
from qe_util import parse_qe_input, parse_qe_forces, run_espresso, \
    edit_qe_input_positions, qe_input_to_structure


def cleanup_espresso_run(target: str = None):
    os.system('rm pwscf.out')
    os.system('rm pwscf.wfc')
    os.system('rm pwscf.save')
    os.system('rm pwscf.in')
    os.system('rm pwscf.wfc1')
    if target:
        os.system('rm ' + target)


# ------------------------------------------------------
#                   test  otf helper functions
# ------------------------------------------------------


@pytest.mark.parametrize("qe_input,exp_pos",
                         [
                             ('./test_files/qe_input_1.in',
                              [np.array([2.51857, 2.5, 2.5]),
                               np.array([4.48143, 2.5, 2.5])])
                         ]
                         )
def test_position_parsing(qe_input, exp_pos):
    positions, species, cell, masses = parse_qe_input(qe_input)
    assert len(positions) == len(exp_pos)

    for i, pos in enumerate(positions):
        assert np.all(pos == exp_pos[i])


@pytest.mark.parametrize("qe_input,exp_spec",
                         [
                             ('./test_files/qe_input_1.in',
                              ['H', 'H'])
                         ]
                         )
def test_species_parsing(qe_input, exp_spec):
    positions, species, cell, masses = parse_qe_input(qe_input)
    assert len(species) == len(exp_spec)
    for i, spec in enumerate(species):
        assert spec == exp_spec[i]


@pytest.mark.parametrize("qe_input,exp_cell",
                         [
                             ('./test_files/qe_input_1.in',
                              5.0 * np.eye(3))
                         ]
                         )
def test_cell_parsing(qe_input, exp_cell):
    positions, species, cell, masses = parse_qe_input(qe_input)
    assert np.all(exp_cell == cell)


@pytest.mark.parametrize("qe_input,mass_dict",
                         [
                             ('./test_files/qe_input_1.in',
                              {'H': 0.00010364269933008285}),
                             ('./test_files/qe_input_2.in',
                              {'Al': 0.00010364269933008285 * 26.9815385})
                         ]
                         )
def test_cell_parsing(qe_input, mass_dict):
    positions, species, cell, masses = parse_qe_input(qe_input)
    assert masses == mass_dict


@pytest.mark.parametrize("qe_input",
                         [
                             './test_files/qe_input_1.in',
                             './test_files/qe_input_2.in',
                             './test_files/qe_input_3.in'
                         ]
                         )
def test_input_to_structure(qe_input):
    assert isinstance(qe_input_to_structure(qe_input), Structure)


@pytest.mark.parametrize('qe_output,exp_forces',
                         [
                             ('./test_files/qe_output_1.out',
                              [np.array([1.90627356, 0., 0.]),
                               np.array([-1.90627356, 0., 0.])])
                         ]
                         )
def test_force_parsing(qe_output, exp_forces):
    forces = parse_qe_forces(qe_output)
    assert len(forces) == len(exp_forces)

    for i, force in enumerate(forces):
        assert np.isclose(force, exp_forces[i]).all()


@pytest.mark.parametrize('qe_input,qe_output',
                         [
                             ('./test_files/qe_input_1.in',
                              './test_files/qe_output_1.out')
                         ]
                         )
def test_espresso_calling(qe_input, qe_output):
    assert os.environ.get('PWSCF_COMMAND',
                          False), 'PWSCF_COMMAND not found ' \
                                  'in environment'

    pw_loc = os.environ.get('PWSCF_COMMAND')
    os.system(' '.join(['cp', qe_input, 'pwscf.in']))
    positions, species, cell, masses = parse_qe_input(qe_input)

    struc = Structure(cell=cell, species=species, positions=positions,
                      mass_dict=masses)

    forces = run_espresso('pwscf.in',
                          struc, pw_loc)

    ref_forces = parse_qe_forces(qe_output)

    assert len(forces) == len(ref_forces)

    for i in range(struc.nat):
        assert np.isclose(forces[i], ref_forces[i]).all()

    cleanup_espresso_run()


def test_espresso_input_edit():
    """
    Load a structure in from qe_input_1, change the position and cell,
    then edit and re-parse
    :return:
    """
    os.system('cp test_files/qe_input_1.in .')
    positions, species, cell, masses = parse_qe_input('./qe_input_1.in')
    struc = Structure(cell, species, positions, masses)

    struc.vec1 += np.random.randn(3)
    struc.positions[0] += np.random.randn(3)

    edit_qe_input_positions('./qe_input_1.in', structure=struc)

    positions, species, cell, masses = parse_qe_input('./qe_input_1.in')

    assert np.equal(positions[0], struc.positions[0]).all()
    assert np.equal(struc.vec1, cell[0, :]).all()

    os.system('rm ./qe_input_1.in')
