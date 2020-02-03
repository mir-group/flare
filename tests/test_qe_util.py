import pytest
import os
import sys
import numpy as np
from flare.struc import Structure, get_unique_species
from flare.dft_interface.qe_util import parse_dft_input, parse_dft_forces, run_dft_par, \
    edit_dft_input_positions, dft_input_to_structure

def cleanup_espresso_run(basename: str, target: str = None):
    os.system(f'rm {basename}.out')
    os.system(f'rm {basename}.wfc')
    os.system(f'rm -r {basename}.save')
    os.system(f'rm {basename}.in')
    os.system(f'rm {basename}.wfc1')
    os.system(f'rm {basename}.wfc2')
    os.system(f'rm {basename}.xml')
    if target:
        os.system(f'rm {target}')


# ------------------------------------------------------
#                   test  otf helper functions
# ------------------------------------------------------

@pytest.mark.parametrize("qe_input,exp_spec",
                         [
                             ('./test_files/qe_input_1.in',
                              ['H', 'H'])
                         ]
                         )
def test_species_parsing(qe_input, exp_spec):
    positions, species, cell, masses = parse_dft_input(qe_input)
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
    positions, species, cell, masses = parse_dft_input(qe_input)
    assert np.all(exp_cell == cell)


# @pytest.mark.parametrize("qe_input,mass_dict",
#                          [
#                              ('./test_files/qe_input_1.in',
#                               {'H': 0.00010364269933008285}),
#                              ('./test_files/qe_input_2.in',
#                               {'Al': 0.00010364269933008285 * 26.9815385})
#                          ]
#                          )
# def test_cell_parsing(qe_input, mass_dict):
#     positions, species, cell, masses = parse_dft_input(qe_input)
#     assert masses == mass_dict


@pytest.mark.parametrize("qe_input",
                         [
                             './test_files/qe_input_1.in',
                             './test_files/qe_input_2.in',
                             './test_files/qe_input_3.in'
                         ]
                         )
def test_input_to_structure(qe_input):
    assert isinstance(dft_input_to_structure(qe_input), Structure)


@pytest.mark.parametrize('qe_input,qe_output',
                         [
                             ('./test_files/qe_input_1.in',
                              './test_files/qe_output_1.out')
                         ]
                         )
@pytest.mark.skipif(not os.environ.get('PWSCF_COMMAND',
                          False), reason='PWSCF_COMMAND not found '
                                  'in environment: Please install Quantum '
                                  'ESPRESSO and set the PWSCF_COMMAND env. '
                                  'variable to point to pw.x.')
def test_espresso_calling(qe_input, qe_output):

    dft_loc = os.environ.get('PWSCF_COMMAND')
    os.system(' '.join(['cp', qe_input, 'pwscf.in']))
    positions, species, cell, masses = parse_dft_input(qe_input)

    structure = Structure(cell=cell, species=species,
                          positions=positions,
                          mass_dict=masses, species_labels=species)

    forces = run_dft_par('pwscf.in', structure, dft_loc, dft_out='pwscf.out')

    ref_forces = parse_dft_forces('pwscf.out')

    assert len(forces) == len(ref_forces)

    for i in range(structure.nat):
        assert np.allclose(forces[i], ref_forces[i])

    cleanup_espresso_run('pwscf')


def test_espresso_input_edit():
    """
    Load a structure in from qe_input_1, change the position and cell,
    then edit and re-parse
    :return:
    """
    os.system('cp test_files/qe_input_1.in .')
    positions, species, cell, masses = parse_dft_input('./qe_input_1.in')
    _, coded_species = get_unique_species(species)
    structure = Structure(cell, coded_species, positions, masses,
                          species_labels=species)

    structure.vec1 += np.random.randn(3)
    structure.positions[0] += np.random.randn(3)

    new_file = edit_dft_input_positions('./qe_input_1.in', structure=structure)

    positions, species, cell, masses = parse_dft_input(new_file)

    assert np.equal(positions[0], structure.positions[0]).all()
    assert np.equal(structure.vec1, cell[0, :]).all()

    os.system('rm ./qe_input_1.in')
