from flare.struc import Structure
import pymatgen as pmg
from copy import deepcopy
import numpy as np
from typing import List, Union

from flare.utils.element_coder import element_to_Z




def config_step(init_struc) -> Structure:
    pass


def trivial_accept(*args, **kwargs) -> bool:
    return True


def baby_accept_reject(new_struc, init_struc=None) -> bool:
    pass


def eval_energy(new_struc) -> float:
    pass


def energy_accept_reject(new_struc, energy) -> bool:
    # Metropoliz algorithm -- Boltzmann factor - need a beta value (see 275 Notes)
    # CS 124 PP3?
    pass


class MCStructureModulator:
    def __init__(self, *args, **kwargs):
        pass

    def modulate_structure(self, structure, target_atoms: List[Union[int, str]] = None):
        raise NotImplementedError


class MCStructureGenerator:
    def __init__(
        self, base_structure: Structure,
            modulators: List[MCStructureModulator]
    ):
        self.orig_structure = deepcopy(base_structure)
        self.structure = base_structure
        self.modulators = modulators

    def get_next_structure(self,
                           structure)->Structure:
        new_struc = structure

        for modulate in self.modulators:
            new_struc = modulate.modulate_structure(new_struc)

        return new_struc




class MCAtomSwapper(MCStructureModulator):
    def __init__(self, swap_probs: dict):
        """

        Args:
            swap_probs: Dictionary with key= from element, value as dictionary,
                which contains key = to element, value = transition probability.
                Transition probabilities must sum to 1.
                Example: {'H':{'H':0.5, 'He':0.5}} will make a H atom have a 50-50
                probability of becoming a He or H atom.
        """

        self.swap_probs = swap_probs

        for transition_prob in swap_probs.values():
            sum = np.sum(list(transition_prob.values()))
            assert sum == 1.0, "Transition probabilities do not sum to 1!"

        pass

    def modulate_structure(self, structure, target_atoms: List[Union[int, str]] = None):
        """
        Swap species based on odds-dictionary describing a set of probabilities.
        Target atoms can be either a list of integers, describing which atoms to target,
        or a list of species.
        Args:
            structure:
            target_atoms:
            swap_odds:

        Returns:
        """

        if target_atoms is None:
            target_atoms = list(range(len(structure)))

        new_struc = deepcopy(structure)

        for i in target_atoms:
            cur_elt = structure.species_labels[i]

            if cur_elt in self.swap_probs.keys():

                # Obtain new species
                items = self.swap_probs[cur_elt].items()
                output_elts = [item[0] for item in items]
                probs = [item[1] for item in items]
                new_species = np.random.choice(output_elts, p=probs)
                # Assign it
                new_struc.species_labels[i] = new_species
                new_struc.coded_species[i] = element_to_Z(new_species)

        return new_struc
