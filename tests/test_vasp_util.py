import pytest
import os
import sys
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Vasprun
from flare.struc import Structure, get_unique_species
from flare.dft_interface.vasp_util import parse_dft_forces, run_dft, \
    edit_dft_input_positions, dft_input_to_structure

def cleanup_vasp_run(target: str = None):
    os.system('rm POSCAR')
    os.system('rm vasprun.xml')

# ------------------------------------------------------
#                   test  otf helper functions
# ------------------------------------------------------

@pytest.mark.parametrize("poscar",
                         [
                             ('./test_files/test_POSCAR')
                         ]
                         )
def test_structure_parsing(poscar):
    structure = dft_input_to_structure(poscar)
    pmg_struct = Poscar.from_file(poscar).structure
    assert len(structure.species_labels) == len(pmg_struct)
    assert structure.cell == pmg_struct.lattice.matrix
    for i, spec in enumerate(structure.species_labels):
        assert spec == pmg_struc[i].specie.Z

@pytest.mark.parametrize("poscar",
                         [
                             './test_files/test_POSCAR'
                         ]
                         )
def test_input_to_structure(poscar):
    assert isinstance(dft_input_to_structure(poscar), Structure)    

@pytest.mark.parametrize('cmd', 'poscar',
                         [
                             ('./test_files/dummy_vasp.py',
                              './test_files/test_POSCAR')
                         ]
                         )
def test_vasp_calling(cmd, poscar):

    cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)), cmd)

    structure = dft_input_to_structure(poscar)

    forces = run_dft('.', cmd, structure=structure, en=False)

    ref_forces = parse_dft_forces(qe_output)

    assert len(forces) == len(ref_forces)

    for i in range(structure.nat):
        assert np.isclose(forces[i], ref_forces[i]).all()

    cleanup_vasp_run()


def test_espresso_input_edit():
    os.system('cp test_files/test_POSCAR ./POSCAR')
    structure = dft_input_to_structure('./test_files/POSCAR')

    structure.vec1 += np.random.randn(3)
    structure.positions[0] += np.random.randn(3)

    new_file = edit_dft_input_positions('./POSCAR', structure=structure)

    final_structure = dft_input_to_structure(new_file)

    assert np.equal(positions[0], structure.positions[0]).all()
    assert np.equal(structure.vec1, cell[0, :]).all()

    os.system('rm ./POSCAR')
