"""
Utility functions for various tasks.
"""
from warnings import warn
from json import JSONEncoder
from typing import List
from math import inf

import numpy as np


def get_random_velocities(noa: int, temperature: float, mass: float):
    """Draw velocities from the Maxwell-Boltzmann distribution, assuming a
    fixed mass for all particles in amu.

    Args:
        noa (int): Number of atoms in the system.
        temperature (float): Temperature of the system.
        mass (float): Mass of each particle in amu.

    Returns:
        np.ndarray: Particle velocities, corrected to give zero center of mass motion.
    """
    
    # Use FLARE mass units (time = ps, length = A, energy = eV)
    mass_md = mass * 0.000103642695727
    kb = 0.0000861733034
    std = np.sqrt(kb * temperature / mass_md)
    velocities = np.random.normal(scale=std, size=(noa, 3))
    
    # Remove center-of-mass motion
    vel_sum = np.sum(velocities, axis=0)
    corrected_velocities = velocities - vel_sum / noa

    return corrected_velocities


def multicomponent_velocities(temperature: float, masses: List[float]):
    """Draw velocities from the Maxwell-Boltzmann distribution for particles of
    varying mass.

    Args:
        temperature (float): Temperature of the system.
        masses (List[float]): Particle masses in amu.

    Returns:
        np.ndarray: Particle velocities, corrected to give zero center of mass motion.
    """

    noa = len(masses)
    velocities = np.zeros((noa, 3))
    kb = 0.0000861733034
    mom_tot = np.array([0., 0., 0.])

    for n, mass in enumerate(masses):
        # Convert to FLARE mass units (time = ps, length = A, energy = eV)
        mass_md = mass * 0.000103642695727
        std = np.sqrt(kb * temperature / mass_md)
        rand_vel = np.random.normal(scale=std, size=(3))
        velocities[n] = rand_vel
        mom_curr = rand_vel * mass_md
        mom_tot += mom_curr

    # Correct momentum, remove center of mass motion
    mom_corr = mom_tot / noa
    for n, mass in enumerate(masses):
        mass_md = mass * 0.000103642695727
        velocities[n] -= mom_corr / mass_md

    return velocities


def get_supercell_positions(sc_size: int, cell: np.ndarray,
                            positions: np.ndarray):
    """Returns the positions of a supercell of atoms, with the number of cells
    in each direction fixed.

    Args:
        sc_size (int): Size of the supercell.
        cell (np.ndarray): 3x3 array of cell vectors.
        positions (np.ndarray): Positions of atoms in the unit cell.

    Returns:
        np.ndarray: Positions of atoms in the supercell.
    """

    sc_positions = []
    for m in range(sc_size):
        vec1 = m * cell[0]
        for n in range(sc_size):
            vec2 = n * cell[1]
            for p in range(sc_size):
                vec3 = p * cell[2]

                # append translated positions
                for pos in positions:
                    sc_positions.append(pos+vec1+vec2+vec3)

    return np.array(sc_positions)


def supercell_custom(cell: np.ndarray, positions: np.ndarray,
                     size1: int, size2: int, size3: int):
    """Returns the positions of a supercell of atoms with a chosen number of
    cells in each direction.

    Args:
        cell (np.ndarray): 3x3 array of cell vectors.
        positions (np.ndarray): Positions of atoms in the unit cell.
        size1 (int): Number of cells along the first cell vector.
        size2 (int): Number of cells along the second cell vector.
        size3 (int): Number of cells along the third cell vector.

    Returns:
        np.ndarray: Positions of atoms in the supercell.
    """

    sc_positions = []
    for m in range(size1):
        vec1 = m * cell[0]
        for n in range(size2):
            vec2 = n * cell[1]
            for p in range(size3):
                vec3 = p * cell[2]

                # append translated positions
                for pos in positions:
                    sc_positions.append(pos+vec1+vec2+vec3)

    return np.array(sc_positions)


# Dictionary mapping elements to their atomic number (Z)
_element_to_Z = {'H': 1,
                 'He': 2,
                 'Li': 3,
                 'Be': 4,
                 'B': 5,
                 'C': 6,
                 'N': 7,
                 'O': 8,
                 'F': 9,
                 'Ne': 10,
                 'Na': 11,
                 'Mg': 12,
                 'Al': 13,
                 'Si': 14,
                 'P': 15,
                 'S': 16,
                 'Cl': 17,
                 'Ar': 18,
                 'K': 19,
                 'Ca': 20,
                 'Sc': 21,
                 'Ti': 22,
                 'V': 23,
                 'Cr': 24,
                 'Mn': 25,
                 'Fe': 26,
                 'Co': 27,
                 'Ni': 28,
                 'Cu': 29,
                 'Zn': 30,
                 'Ga': 31,
                 'Ge': 32,
                 'As': 33,
                 'Se': 34,
                 'Br': 35,
                 'Kr': 36,
                 'Rb': 37,
                 'Sr': 38,
                 'Y': 39,
                 'Zr': 40,
                 'Nb': 41,
                 'Mo': 42,
                 'Tc': 43,
                 'Ru': 44,
                 'Rh': 45,
                 'Pd': 46,
                 'Ag': 47,
                 'Cd': 48,
                 'In': 49,
                 'Sn': 50,
                 'Sb': 51,
                 'Te': 52,
                 'I': 53,
                 'Xe': 54,
                 'Cs': 55,
                 'Ba': 56,
                 'La': 57,
                 'Ce': 58,
                 'Pr': 59,
                 'Nd': 60,
                 'Pm': 61,
                 'Sm': 62,
                 'Eu': 63,
                 'Gd': 64,
                 'Tb': 65,
                 'Dy': 66,
                 'Ho': 67,
                 'Er': 68,
                 'Tm': 69,
                 'Yb': 70,
                 'Lu': 71,
                 'Hf': 72,
                 'Ta': 73,
                 'W': 74,
                 'Re': 75,
                 'Os': 76,
                 'Ir': 77,
                 'Pt': 78,
                 'Au': 79,
                 'Hg': 80,
                 'Tl': 81,
                 'Pb': 82,
                 'Bi': 83,
                 'Po': 84,
                 'At': 85,
                 'Rn': 86,
                 'Fr': 87,
                 'Ra': 88,
                 'Ac': 89,
                 'Th': 90,
                 'Pa': 91,
                 'U': 92,
                 'Np': 93,
                 'Pu': 94,
                 'Am': 95,
                 'Cm': 96,
                 'Bk': 97,
                 'Cf': 98,
                 'Es': 99,
                 'Fm': 100,
                 'Md': 101,
                 'No': 102,
                 'Lr': 103,
                 'Rf': 104,
                 'Db': 105,
                 'Sg': 106,
                 'Bh': 107,
                 'Hs': 108,
                 'Mt': 109,
                 'Ds': 110,
                 'Rg': 111,
                 'Cn': 112,
                 'Nh': 113,
                 'Fl': 114,
                 'Mc': 115,
                 'Lv': 116,
                 'Ts': 117,
                 'Og': 118}

# Define inverse mapping
_Z_to_element = {z: elt for elt, z in _element_to_Z.items()}


def element_to_Z(element: str) -> int:
    """
    Returns the atomic number Z associated with an elements 1-2 letter name.
    Returns the same integer if an integer is passed in.

    :param element:
    :return:
    """

    # If already integer, do nothing
    if isinstance(element, (int, np.integer)):
        return element
    if type(element).__module__ == 'numpy' and np.issubdtype(type(element),
                                                             np.integer):
        return element

    # If a string-casted integer, do nothing
    if isinstance(element, str) and element.isnumeric():
        return int(element)

    # Check that a valid element was passed in then return
    if _element_to_Z.get(element, None) is None:
        warn('Element as specified not found in list of element-Z mappings. '
             'If you would like to specify a custom element, use an integer '
             'of your choosing instead. Setting element {} to integer '
             '0'.format(element))
    return _element_to_Z.get(element, 0)


class NumpyEncoder(JSONEncoder):
    """
    Special json encoder for numpy types for serialization
    use as

    json.loads(... cls = NumpyEncoder)

    or:

    json.dumps(... cls = NumpyEncoder)

    Thanks to StackOverflow users karlB and fnunnari, who contributed this from:
    `https://stackoverflow.com/a/47626762`
    """

    def default(self, obj):
        """
        """
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def Z_to_element(Z: int) -> str:
    """
    Maps atomic numbers Z to element name, e.g. 1->"H".

    :param Z: Atomic number corresponding to element.
    :return: One or two-letter name of element.
    """

    # Check proper formatting
    if isinstance(Z, str):
        if Z.isnumeric():
            Z = int(Z)
        else:
            raise ValueError("Input Z is not a number. It should be an "
                             "integer")
    return _Z_to_element[Z]


def is_std_in_bound(std_tolerance: float, noise: float,
                    structure: 'flare.struc.Structure',
                    max_atoms_added: int = inf)-> (bool, List[int]):
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
                                max_by_species: dict = {})-> (bool, List[int]):
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
    max_std_components = [np.max(std) for std in structure.stds]
    if max(max_std_components) < threshold:
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
                max_std_components[i] < threshold:
            break

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
                                  max_by_species: dict ={},
                                  max_force_error: float
                                  = inf)-> (bool, List[int]):
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
    if np.max(max_error_components) < abs_force_tolerance:
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
                max_error_components[i] < abs_force_tolerance:
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
