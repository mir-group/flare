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

def element_to_Z(element:str)->int:
    """
    Returns the atomic number Z associated with an elements 1-2 letter name.
    Returns the same integer if an integer is passed in.
    :param element:
    :return:
    """

    # If already integer, do nothing
    if isinstance(element, int):
        return element

    if type(element).__module__ == 'numpy' and np.issubdtype(element,
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

def Z_to_element(Z: int)-> str:

    if isinstance(Z,str):
        if Z.isnumeric():
            Z = int(Z)
        else:
            raise ValueError("Input Z is not a number. It should be an "
                             "integer")
    return _Z_to_element[Z]
