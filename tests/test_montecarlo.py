from flare.montecarlo import MCStructureGenerator
from flare.struc import Structure
import numpy as np

from flare.montecarlo import MCAtomSwapper


def test_atom_swap():

    test_struc = Structure(cell=np.eye(3), species=["H"], positions=[[0, 0, 0]])

    trivial_swap_dict = {"H": {"H": 1.0}}

    mc_swapper = MCAtomSwapper(trivial_swap_dict)

    new_struc = mc_swapper.modulate_structure(test_struc)

    assert new_struc.species_labels[0] == test_struc.species_labels[0]
    assert new_struc.coded_species[0] == test_struc.coded_species[0]

    helium_swap_dict = {"H": {"He": 1.0}}

    mc_swapper = MCAtomSwapper(helium_swap_dict)

    new_struc = mc_swapper.modulate_structure(test_struc)

    assert new_struc.species_labels[0] == "He"



def test_generator():
    pass
