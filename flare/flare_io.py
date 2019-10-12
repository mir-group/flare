from flare.struc import Structure
from typing import List
from json import dump, load
from flare.util import NumpyEncoder

def md_trajectory_to_file(filename, structures: List[Structure]):
	"""
	Take a list of structures and write them to a json file.
	:param filename:
	:param structures:
	"""
	f = open(filename, 'w')
	dump([s.as_dict() for s in structures], f, cls=NumpyEncoder)
	f.close()

def md_trajectory_from_file(filename):
	"""
	Read a list of structures from a json file, formatted as in md_trajectory_to_file.
	:param filename:
	"""
	f = open(filename, 'r')
	structure_list = load(f)
	structures = [Structure.from_dict(dictionary) for dictionary in structure_list]
	return structures
