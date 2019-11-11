"""
Utility functions for various tasks

2019
"""
from warnings import warn
import numpy as np
from json import JSONEncoder

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

    if isinstance(element, str) and element.isnumeric():
        return int(element)

    if _element_to_Z.get(element, None) is None:
        warn('Element as specified not found in list of element-Z mappings. '
             'If you would like to specify a custom element, use an integer '
             'of your choosing instead. Setting element {} to integer '
             '0'.format(element))
    return _element_to_Z.get(element, 0)


class NumpyEncoder(JSONEncoder):
    """
    Special json encoder for numpy types for serialization
    use as  json.loads(... cls = NumpyEncoder)
    or json.dumps(... cls = NumpyEncoder)
    Thanks to StackOverflow users karlB and fnunnari
    https://stackoverflow.com/a/47626762
    """

    def default(self, obj):
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
    if isinstance(Z, str):
        if Z.isnumeric():
            Z = int(Z)
        else:
            raise ValueError("Input Z is not a number. It should be an "
                             "integer")
    return _Z_to_element[Z]


def is_std_in_bound(std_tolerance, noise, structure, max_atoms_added):
    # set uncertainty threshold
    if std_tolerance == 0:
        return True, [-1]
    elif std_tolerance > 0:
        threshold = std_tolerance * np.abs(noise)
    else:
        threshold = np.abs(std_tolerance)

    # sort max stds
    nat = structure.nat
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
                                structure, max_atoms_added: int =
                                np.inf, max_by_species: dict = {}):
    """
    Checks the stds of GP prediction assigned to the structure, returns a
    list of atoms which either meet an absolute threshold or a relative
    threshold defined by rel_std_tolerance * noise. Can limit the
    total number of target atoms via max_atoms_added, and limit per species
    by max_by_species.

    The max_atoms_added argument will 'overrule' the
    max by species; e.g. if max_atoms_added is 2 and max_by_species is {"H":3},
    then at most two atoms total will be added.

    :param rel_std_tolerance:
    :param abs_std_tolerance:
    :param noise:
    :param structure:
    :param max_atoms_added:
    :param max_by_species:
    :return:
    """

    # This indicates test mode, as the GP is not being modified in any way
    if rel_std_tolerance == 0 and abs_std_tolerance == 0:
        return True, [-1]

    # set uncertainty threshold
    if rel_std_tolerance is None or rel_std_tolerance == 0:
        threshold = abs_std_tolerance
    elif abs_std_tolerance is None or abs_std_tolerance == 0:
        threshold = rel_std_tolerance * np.abs(noise)
    else:
        threshold = min(rel_std_tolerance * np.abs(noise),
                        abs_std_tolerance)

    # Determine if any std component will trigger the threshold
    max_std_components = [np.max(std) for std in structure.stds]
    if max(max_std_components) < threshold:
        return True, [-1]

    target_atoms = []

    # Sort from greatest to smallest max. std component
    std_arg_sorted = np.flip(np.argsort(max_std_components))

    present_species = {spec: 0 for spec in set(structure.species_labels)}

    # Only add atoms up to the bound
    for i in std_arg_sorted:

        # If max atoms added reached or stds are now below threshold, done
        if len(target_atoms) == max_atoms_added or \
                max_std_components[i] < threshold:
            break

        cur_spec = structure.species_labels[i]

        # Only add up to species allowance, if it exists
        if present_species[cur_spec] < \
                max_by_species.get(cur_spec, np.inf):
            target_atoms.append(i)
            present_species[cur_spec] += 1

    return False, target_atoms


def is_force_in_bound_per_species(abs_force_tolerance: float,
                                  predicted_forces: np.array,
                                  label_forces: np.array,
                                  structure,
                                  max_atoms_added: int = np.inf,
                                  max_by_species: dict =
                                  {}, max_force_error: float = np.inf):
    """
    Checks the forces of GP prediction assigned to the structure, returns a
    list of atoms which  meet an absolute threshold abs_force_tolerance. Can 
    limit the total number of target atoms via max_atoms_added, and limit 
    per species by max_by_species.

    The max_atoms_added argument will 'overrule' the
    max by species; e.g. if max_atoms_added is 2 and max_by_species is {"H":3},
    then at most two atoms total will be added.

    :param abs_force_tolerance:
    :param guesses:
    :param labels:
    :param structure:
    :param max_atoms_added:
    :param max_by_species:
    :param max_force_error: In order to avoid counting in highly unlikely
    configurations, if the error exceeds this, do not add atom
    :return:
    """

    # This indicates test mode, as the GP is not being modified in any way
    if abs_force_tolerance == 0:
        return True, [-1]

    errors = np.abs(predicted_forces - label_forces)
    # Determine if any std component will trigger the threshold
    max_error_components = np.amax(errors, axis=1)

    if np.max(max_error_components) < abs_force_tolerance:
        return True, [-1]

    target_atoms = []

    # Sort from greatest to smallest max. std component
    force_arg_sorted = np.flip(np.argsort(max_error_components))

    present_species = {spec: 0 for spec in set(structure.species_labels)}

    # Only add atoms up to the bound
    for i in force_arg_sorted:

        # If max atoms added reached or forces are now below threshold, done
        if len(target_atoms) == max_atoms_added or \
                max_error_components[i] < abs_force_tolerance:
            break

        cur_spec = structure.species_labels[i]

        # Only add up to species allowance, if it exists
        if present_species[cur_spec] < \
                max_by_species.get(cur_spec, np.inf) \
                and max_error_components[i] < max_force_error:
            target_atoms.append(i)
            present_species[cur_spec] += 1

    # Handle the case that nothing was added
    if len(target_atoms):
        return False, target_atoms
    else:
        return True, [-1]

