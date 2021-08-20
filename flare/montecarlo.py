from flare.struc import Structure
import pymatgen as pmg
from copy import deepcopy
import numpy as np
from typing import List, Union

from flare.utils.element_coder import element_to_Z


def eval_energy(structure) -> float:
    pass


class MCAcceptor:
    def __init__(self, *args, **kwargs):
        pass

    def accept_reject(self, structure):
        raise NotImplementedError


class MCTrivialAcceptor(MCAcceptor):
    def __init__(self, *args, **kwargs):
        pass

    def accept_reject(self, structure) -> bool:
        """
        Trivially accepts a given structure.
        Args:
            structure:

        Returns:

        """
        return True


class MCInitAcceptor(MCAcceptor):
    def __init__(self, *args, **kwargs):
        pass

    def accept_reject(self, structure, prev_structure=None) -> bool:
        """
        Accepts or rejects (boolaen) a given structure relative to previous iteration of the structure.
        Args:
            structure:
            prev_structure:

        Returns:

        """
        raise NotImplementedError


class MCEnergyAcceptor(MCAcceptor):
    def __init__(self, *args, **kwargs):
        pass

    def accept_reject(self, structure, energy) -> bool:
        """
        Accepts or rejects (boolean) a given structure based on its energy.
        Args:
            structure:
            energy:

        Returns:

        """
        # Metropolis algo -- Boltzmann factor - need a beta value (see 275 Notes)
        raise NotImplementedError


class MCStructureEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate_structure(self, structure, acceptance: bool):
        """
        Evaluate structure based on acceptance boolean.
        Args:
            structure:
            acceptance:

        Returns:

        """
        raise NotImplementedError


class MCStructureModulator:
    def __init__(self, *args, **kwargs):
        pass

    def modulate_structure(self, structure, target_atoms: List[Union[int, str]] = None):
        raise NotImplementedError


class MCStructureGenerator:
    def __init__(
        self, base_structure: Structure, modulators: List[MCStructureModulator]
    ):
        self.orig_structure = deepcopy(base_structure)
        self.structure = base_structure
        self.modulators = modulators

    def get_next_structure(self, structure) -> Structure:
        new_struc = structure

        for modulate in self.modulators:
            new_struc = modulate.modulate_structure(new_struc)

        return new_struc

    def __nonzero__(self):
        return len(self.modulators)


class MCAtomSwapper(MCStructureModulator):
    def __init__(self, swap_probs: dict, swap_num: int = 0, swap_prob: float = 0):
        """

        Args
            swap_probs: Dictionary with key = from element, value as dictionary,
                which contains key = to element, value = swap probability.
                Transition probabilities must sum to 1.
                Example: {'H_2':{'H_1':0.5, 'He_3':0.5}} will make a H atom at index 2 have a 50-50
                probability of swapping with the H atom at index or He atom at index 3.

            #     Alternative handling of swapping: swap_targets: Dictionary with key = from element,
            #         value as list of target elements that the element can be swapped with.

            swap_num: Number of swaps to be performed.
            swap_prob: Probability a given pair of elements will be swapped.
        """

        self.swap_probs = swap_probs
        self.swap_num = swap_num
        self.swap_prob = swap_prob

        if swap_num == 0 and swap_prob == 0:
            raise ValueError("Make swap_num or swap_prob nonzero.")

    def modulate_structure(self, structure, target_atoms: List[Union[int, str]] = None):
        """
        Swap species with another element in the structure.
        Concentration of each element in the configuration is conserved.
        Args:
            structure:
            target_atoms: List of integers, describing which atoms to target, or a list of species.

        Returns: new_struc

        """
        if target_atoms is None:
            target_atoms = list(range(len(structure)))

        new_struc = deepcopy(structure)

        for _ in range(self.swap_num):
            # Obtain first element in swap
            from_elt = np.random.choice(target_atoms)

            if from_elt in self.swap_probs.keys():
                # Obtain to_elt
                items = self.swap_probs[from_elt].items()
                to_elts = [item[0] for item in items]
                probs = [item[1] for item in items]
                to_elt = np.random.choice(to_elts, p=probs)
                # Assign it
                new_struc.reassign_species(from_elt, to_elt)
                new_struc.reassign_species(to_elt, from_elt)

        return new_struc


class MCAtomTransmute(MCStructureModulator):
    def __init__(self, swap_probs: dict):
        """

        Args:
            swap_probs: Dictionary with key = from element, value as dictionary,
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
        Transmute species based on odds-dictionary describing a set of probabilities.
        Args:
            structure:
            target_atoms: List of integers, describing which atoms to target, or a list of species.

        Returns: new_struc
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
                new_struc.reassign_species(i, new_species)

        return new_struc
