from flare.struc import Structure
from typing import List
from json import dump, load
from flare.utils.element_coder import NumpyEncoder

def md_trajectory_to_file(filename: str, structures: List[Structure]):
    """
    Take a list of structures and write them to a json file.
    :param filename:
    :param structures:
    """
    with open(filename, 'w') as f:
        dump([s.as_dict() for s in structures], f, cls=NumpyEncoder)

def md_trajectory_from_file(filename: str):
    """
    Read a list of structures from a json file, formatted as in md_trajectory_to_file.
    :param filename:
    """
    with open(filename, 'r') as f:
        structure_list = load(f)
        structures = \
            [Structure.from_dict(dictionary) for dictionary in structure_list]
    return structures
