import pytest
from os import remove, environ
from shutil import copyfile
import sys
import numpy as np
from flare.struc import Structure, get_unique_species
from flare.dft_interface.cp2k_util import parse_dft_input, parse_dft_forces, run_dft_par, \
    edit_dft_input_positions, dft_input_to_structure


def cleanup(target: str = None):
    for i in ['cp2k-RESTART.wfn', 'dft.out', 'cp2k.in']:
        try:
            remove(i)
        except:
            pass
    if (target is not None):
        remove(target)


@pytest.mark.parametrize("cp2k_input,exp_spec",
                         [
                             ('./test_files/cp2k_input_1.in',
                              ['H', 'H'])
                         ]
                         )
def test_species_parsing(cp2k_input, exp_spec):
    positions, species, cell, masses = parse_dft_input(cp2k_input)
    assert len(species) == len(exp_spec)
    for i, spec in enumerate(species):
        assert spec == exp_spec[i]


@pytest.mark.parametrize("cp2k_input,exp_cell",
                         [
                             ('./test_files/cp2k_input_1.in',
                              5.0 * np.eye(3))
                         ]
                         )
def test_cell_parsing(cp2k_input, exp_cell):
    positions, species, cell, masses = parse_dft_input(cp2k_input)
    assert np.all(exp_cell == cell)

# @pytest.mark.parametrize("cp2k_input,mass_dict",
#                          [
#                              ('./test_files/cp2k_input_1.in',
#                               {'H': 0.00010364269933008285}),
#                              ('./test_files/cp2k_input_2.in',
#                               {'Al': 0.00010364269933008285 * 26.9815385})
#                          ]
#                          )
# def test_cell_parsing(cp2k_input, mass_dict):
#     positions, species, cell, masses = parse_dft_input(cp2k_input)
#     assert masses == mass_dict


@pytest.mark.parametrize("cp2k_input",
                         [
                             './test_files/cp2k_input_1.in',
                             './test_files/cp2k_input_2.in',
                             './test_files/cp2k_input_3.in'
                         ]
                         )
def test_input_to_structure(cp2k_input):
    assert isinstance(dft_input_to_structure(cp2k_input), Structure)


@pytest.mark.parametrize('cp2k_input,cp2k_output',
                         [
                             ('./test_files/cp2k_input_1.in',
                              './test_files/cp2k_output_1.out')
                         ]
                         )
@pytest.mark.skipif(not environ.get('CP2K_COMMAND',
                                    False), reason='CP2K_COMMAND not found '
                    'in environment: Please install CP2K '
                    ' and set the CP2K_COMMAND env. '
                    'variable to point to cp2k.popt')
def test_cp2k_calling(cp2k_input, cp2k_output):
    dft_loc = environ.get('CP2K_COMMAND')
    copyfile(cp2k_input, 'cp2k.in')
    positions, species, cell, masses = parse_dft_input(cp2k_input)

    structure = Structure(cell=cell, species=species,
                          positions=positions,
                          mass_dict=masses, species_labels=species)

    forces = run_dft_par('cp2k.in',
                         structure, dft_loc)

    ref_forces = parse_dft_forces(cp2k_output)

    assert len(forces) == len(ref_forces)

    for i in range(structure.nat):
        assert np.isclose(forces[i], ref_forces[i]).all()

    cleanup()


def test_cp2k_input_edit():
    """
    Load a structure in from cp2k_input_1, change the position and cell,
    then edit and re-parse
    :return:
    """
    positions, species, cell, masses = parse_dft_input(
        './test_files/cp2k_input_1.in')
    _, coded_species = get_unique_species(species)
    structure = Structure(cell, coded_species, positions, masses,
                          species_labels=species)

    structure.vec1 += np.random.randn(3)
    structure.positions[0] += np.random.randn(3)

    newfilename = edit_dft_input_positions(
        './test_files/cp2k_input_1.in', structure=structure)

    positions, species, cell, masses = parse_dft_input(newfilename)

    assert np.equal(positions[0], structure.positions[0]).all()
    assert np.equal(structure.vec1, cell[0, :]).all()

    remove(newfilename)
    cleanup()
