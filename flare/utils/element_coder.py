"""
Utility functions for various tasks.
"""
from warnings import warn
from json import JSONEncoder
from typing import List
from math import inf

import numpy as np

_user_element_to_Z = {}
_user_Z_to_element = {}


def inject_user_definition(element: str, Z: int):
    """ Allow user-defined element. The definition
    will override the default ones from the periodic table.

    Example:

    >>> import flare.utils
    >>> import flare.utils.element_coder as ec
    >>> ec.inject_user_definition('C1', 6)
    >>> ec.inject_user_definition('C2', 7)
    >>> ec.inject_user_definition('H1', 1)
    >>> ec.inject_user_definition('H2', 2)

    This block should be executed before any other
    flare modules are imported. And user has to
    be very careful to not let Z overlap with other
    elements in the system

    :param element: string symbol of the element
    :type element: str
    :param Z: corresponding Z
    :type Z: int
    """
    _user_element_to_Z[element] = Z
    _user_Z_to_element[Z] = element


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

    if element in _user_element_to_Z:
        return _user_element_to_Z[element]

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
        warn(f'Element as specified not found in list of element-Z mappings. '
             'If you would like to specify a custom element, use an integer '
             'of your choosing instead. Setting element {element} to integer '
             '0')
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

    if Z in _user_Z_to_element:
        return _user_Z_to_element[Z]

    # Check proper formatting
    if isinstance(Z, str):
        if Z.isnumeric():
            Z = int(Z)
        else:
            raise ValueError("Input Z is not a number. It should be an "
                             "integer")

    return _Z_to_element[Z]


_Z_to_mass = {1: 1.0079,
              2: 4.0026,
              3: 6.941,
              4: 9.0122,
              5: 10.811,
              6: 12.0107,
              7: 14.0067,
              8: 15.9994,
              9: 18.9984,
              10: 20.1797,
              11: 22.9897,
              12: 24.305,
              13: 26.9815,
              14: 28.0855,
              15: 30.9738,
              16: 32.065,
              17: 35.453,
              19: 39.0983,
              18: 39.948,
              20: 40.078,
              21: 44.9559,
              22: 47.867,
              23: 50.9415,
              24: 51.9961,
              25: 54.938,
              26: 55.845,
              28: 58.6934,
              27: 58.9332,
              29: 63.546,
              30: 65.39,
              31: 69.723,
              32: 72.64,
              33: 74.9216,
              34: 78.96,
              35: 79.904,
              36: 83.8,
              37: 85.4678,
              38: 87.62,
              39: 88.9059,
              40: 91.224,
              41: 92.9064,
              42: 95.94,
              43: 98,
              44: 101.07,
              45: 102.9055,
              46: 106.42,
              47: 107.8682,
              48: 112.411,
              49: 114.818,
              50: 118.71,
              51: 121.76,
              53: 126.9045,
              52: 127.6,
              54: 131.293,
              55: 132.9055,
              56: 137.327,
              57: 138.9055,
              58: 140.116,
              59: 140.9077,
              60: 144.24,
              61: 145,
              62: 150.36,
              63: 151.964,
              64: 157.25,
              65: 158.9253,
              66: 162.5,
              67: 164.9303,
              68: 167.259,
              69: 168.9342,
              70: 173.04,
              71: 174.967,
              72: 178.49,
              73: 180.9479,
              74: 183.84,
              75: 186.207,
              76: 190.23,
              77: 192.217,
              78: 195.078,
              79: 196.9665,
              80: 200.59,
              81: 204.3833,
              82: 207.2,
              83: 208.9804,
              84: 209,
              85: 210,
              86: 222,
              87: 223,
              88: 226,
              89: 227,
              91: 231.0359,
              90: 232.0381,
              93: 237,
              92: 238.0289,
              95: 243,
              94: 244,
              96: 247,
              97: 247,
              98: 251,
              99: 252,
              100: 257,
              101: 258,
              102: 259,
              104: 261,
              103: 262,
              105: 262,
              107: 264,
              106: 266,
              109: 268,
              111: 272,
              108: 277}
