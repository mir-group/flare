"""
Utility functions for various tasks.
"""
from warnings import warn
from json import JSONEncoder
from typing import List
from math import inf

import numpy as np

def is_std_in_bound(std_tolerance: float, noise: float,
                    structure: 'flare.struc.Structure',
                    max_atoms_added: int = inf) -> (bool, List[int]):
    """
    Given an uncertainty tolerance and a structure decorated with atoms,
    species, and associated uncertainties, return those which are above a
    given threshold, agnostic to species.

    If std_tolerance is negative, then the threshold used is the absolute
    value of std_tolerance.

    If std_tolerance is positive, then the threshold used is
    std_tolerance * noise.

    If std_tolerance is 0, then do not check.

    :param std_tolerance: If positive, multiply by noise to get cutoff. If
        negative, use absolute value of std_tolerance as cutoff.
    :param noise: Noise variance parameter
    :param structure: Input structure
    :type structure: FLARE Structure
    :param max_atoms_added: Maximum # of atoms to add
    :return: (True,[-1]) if no atoms are above cutoff, (False,[...]) of the
            top `max_atoms_added` uncertainties
    """
    # set uncertainty threshold
    if std_tolerance == 0:
        return True, [-1]
    elif std_tolerance > 0:
        threshold = std_tolerance * np.abs(noise)
    else:
        threshold = np.abs(std_tolerance)

    # sort max stds
    nat = len(structure)
    max_stds = np.zeros((nat))
    for atom, std in enumerate(structure.stds):
        max_stds[atom] = np.max(std)
    stds_sorted = np.argsort(max_stds)
    target_atoms = list(stds_sorted[-max_atoms_added:])

    # if above threshold, return atom
    if max_stds[stds_sorted[-1]] > threshold:
        return False, target_atoms
    else:
        return True, [-1]


def is_std_in_bound_per_species(rel_std_tolerance: float,
                                abs_std_tolerance: float, noise: float,
                                structure: 'flare.struc.Structure',
                                max_atoms_added: int = inf,
                                max_by_species: dict = {}) -> (bool,
                                                               List[int]):
    """
    Checks the stds of GP prediction assigned to the structure, returns a
    list of atoms which either meet an absolute threshold or a relative
    threshold defined by rel_std_tolerance * noise. Can limit the
    total number of target atoms via max_atoms_added, and limit per species
    by max_by_species.

    The max_atoms_added argument will 'overrule' the
    max by species; e.g. if max_atoms_added is 2 and max_by_species is {"H":3},
    then at most two atoms will be added.

    :param rel_std_tolerance: Multiplied by noise to get a lower
        bound for the uncertainty threshold defined relative to the model.
    :param abs_std_tolerance: Used as an absolute lower bound for the
        uncertainty threshold.
    :param noise: Noise hyperparameter for model, used to define relative
        uncertainty cutoff.
    :param structure: FLARE structure decorated with
        uncertainties in structure.stds.
    :param max_atoms_added: Maximum number of atoms to return from structure.
    :param max_by_species: Dictionary describing maximum number of atoms to
        return by species (e.g. {'H':1,'He':2} will return at most 1 H and 2 He
        atoms.)
    :return: Bool indicating if any atoms exceeded the uncertainty
        threshold, and a list of indices of atoms which did, sorted by their
        uncertainty.
    """

    # Always returns true; use this when you want to test model performance
    # without updating the training set.
    if rel_std_tolerance == 0 and abs_std_tolerance == 0:
        return True, [-1]

    # set uncertainty threshold based on if only one or the other is passed in,
    # and use the lower of the two.

    if rel_std_tolerance is None or rel_std_tolerance == 0:
        threshold = abs_std_tolerance
    elif abs_std_tolerance is None or abs_std_tolerance == 0:
        threshold = rel_std_tolerance * np.abs(noise)
    else:
        threshold = min(rel_std_tolerance * np.abs(noise),
                        abs_std_tolerance)

    # Determine if any std component will trigger the threshold
    # before looking through individual species.
    max_std_components = [np.nanmax(std) for std in structure.stds]
    if np.nanmax(max_std_components) < threshold:
        return True, [-1]

    target_atoms = []

    # Sort from greatest to smallest max. std component
    std_arg_sorted = np.flip(np.argsort(max_std_components))

    present_species = {spec: 0 for spec in set(structure.species_labels)}

    # Loop through atoms and add until cutoffs are met.
    for i in std_arg_sorted:

        # If max atoms added reached or stds of atoms considered are now below
        # threshold, conclude
        if len(target_atoms) == max_atoms_added or \
                (max_std_components[i] < threshold and max_std_components[i]
                 != np.nan):
            break

        if np.isnan(max_std_components[i]):
            continue

        # Only add up to species allowance, if it exists
        cur_spec = structure.species_labels[i]
        if present_species[cur_spec] < \
                max_by_species.get(cur_spec, inf):
            target_atoms.append(i)
            present_species[cur_spec] += 1

    # Check in case that nothing was added, e.g. due to species limitations
    if len(target_atoms):
        return False, target_atoms

    return True, [-1]


def is_force_in_bound_per_species(abs_force_tolerance: float,
                                  predicted_forces: 'ndarray',
                                  label_forces: 'ndarray',
                                  structure,
                                  max_atoms_added: int = inf,
                                  max_by_species: dict = {},
                                  max_force_error: float
                                  = inf) -> (bool, List[int]):
    """
    Checks the forces of GP prediction assigned to the structure against a
    DFT calculation, and return a list of atoms which meet an absolute
    threshold abs_force_tolerance.

    Can limit the total number of target atoms via max_atoms_added, and limit
    per species by max_by_species.

    The max_atoms_added argument will 'overrule' the
    max by species; e.g. if max_atoms_added is 2 and max_by_species is {"H":3},
    then at most two atoms total will be added.

    Because adding atoms which are in configurations which are far outside
    of the potential energy surface may not always be
    desirable, a maximum force error can be passed in; atoms with

    :param abs_force_tolerance: If error exceeds this value, then return
        atom index
    :param predicted_forces: Force predictions made by GP model
    :param label_forces: "True" forces computed by DFT
    :param structure: FLARE Structure
    :param max_atoms_added: Maximum atoms to return
    :param max_by_species: Limit to a maximum number of atoms by species
    :param max_force_error: In order to avoid counting in highly unlikely
        configurations, if the error exceeds this, do not add atom
    :return: Bool indicating if any atoms exceeded the error
        threshold, and a list of indices of atoms which did sorted by their
        error.
    """

    # Always returns true; use this when you want to test model performance
    # without updating the training set.
    if abs_force_tolerance == 0:
        return True, [-1]

    errors = np.abs(predicted_forces - label_forces)

    # Determine if any force component will trigger the threshold
    max_error_components = np.amax(errors, axis=1)
    if np.nanmax(max_error_components) < abs_force_tolerance:
        return True, [-1]

    target_atoms = []

    # Sort from greatest to smallest error
    force_arg_sorted = np.flip(np.argsort(max_error_components))

    present_species = {spec: 0 for spec in set(structure.species_labels)}

    # Only add atoms up to the bound
    for i in force_arg_sorted:

        # If max atoms added reached or force errors are now below threshold,
        # conclude
        if len(target_atoms) == max_atoms_added or \
                (max_error_components[i] < abs_force_tolerance and
                 max_error_components[i] != np.nan):
            break

        cur_spec = structure.species_labels[i]

        # Only add up to species allowance, if it exists
        if present_species[cur_spec] < \
                max_by_species.get(cur_spec, inf) \
                and max_error_components[i] < max_force_error:
            target_atoms.append(i)
            present_species[cur_spec] += 1

    # Check in case that nothing was added e.g. due to species or error
    # limitations
    if len(target_atoms):
        return False, target_atoms
    else:
        return True, [-1]


def subset_of_frame_by_element(frame: 'flare.Structure',
                               predict_atoms_per_element: dict) -> List[int]:
    """
    Given a structure and a dictionary formatted as {"Symbol":int,
    ..} describing a number of atoms per element, return a sorted list of
    indices corresponding to a random subset of atoms by species
    :param frame:
    :param predict_atoms_by_species:
    :return:
    """

    # Null case: No dictionary or empty dict passed in; just return all indices
    if not predict_atoms_per_element:
        return list(range(len(frame)))

    # Keep track of atoms which were considered (dictionary may only cover a
    #  subset of species of the whole frame)
    all_atoms = set(range(len(frame)))
    return_atoms = []
    considered_atoms = set([])

    species = frame.species_labels

    # Main loop: Obtain the number of relevant atoms for each element
    for elt, n in predict_atoms_per_element.items():

        matching_atoms = [i for i in all_atoms if species[i] == elt]
        considered_atoms.update(matching_atoms)

        if len(matching_atoms) == 0:
            continue
        # Choose the atoms to add
        to_add_atoms = np.random.choice(matching_atoms, replace=False,
                                        size=min(n, len(matching_atoms)))
        return_atoms += list(to_add_atoms)

    return_atoms += list(all_atoms - considered_atoms)

    return_atoms.sort()

    return return_atoms


def get_max_cutoff(cell: np.ndarray) -> float:
    """Compute the maximum cutoff compatible with a 3x3x3 supercell of a
        structure. Called in the Structure constructor when
        setting the max_cutoff attribute, which is used to create local
        environments with arbitrarily large cutoff radii.

    Args:
        cell (np.ndarray): Bravais lattice vectors of the structure stored as
            rows of a 3x3 Numpy array.

    Returns:
        float: Maximum cutoff compatible with a 3x3x3 supercell of the
            structure.
    """

    # Retrieve the lattice vectors.
    a_vec = cell[0]
    b_vec = cell[1]
    c_vec = cell[2]

    # Compute dot products and norms of lattice vectors.
    a_dot_b = np.dot(a_vec, b_vec)
    a_dot_c = np.dot(a_vec, c_vec)
    b_dot_c = np.dot(b_vec, c_vec)

    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)
    c_norm = np.linalg.norm(c_vec)

    # Compute the six independent altitudes of the cell faces.
    # The smallest is the maximum atomic environment cutoff that can be
    # used with sweep=1.
    max_candidates = np.zeros(6)
    max_candidates[0] = \
        a_norm * np.sqrt(1 - (a_dot_b / (a_norm * b_norm))**2)
    max_candidates[1] = \
        b_norm * np.sqrt(1 - (a_dot_b / (a_norm * b_norm))**2)
    max_candidates[2] = \
        a_norm * np.sqrt(1 - (a_dot_c / (a_norm * c_norm))**2)
    max_candidates[3] = \
        c_norm * np.sqrt(1 - (a_dot_c / (a_norm * c_norm))**2)
    max_candidates[4] = \
        b_norm * np.sqrt(1 - (b_dot_c / (b_norm * c_norm))**2)
    max_candidates[5] = \
        c_norm * np.sqrt(1 - (b_dot_c / (b_norm * c_norm))**2)

    return np.min(max_candidates)
