from flare.struc import Structure
from typing import List
from json import dump, load


def md_trajectory_to_file(filename, structures: List[Structure]):
	f = open(filename, 'w')
	dump([s.as_dict() for s in structures], f)
	f.close()

def md_trajectory_from_file(filename):
	f = open(filename, 'r')
	structure_list = load(f, cls=NumpyEncoder)
	structures = [Structure.from_dict(dictionary) for dictionary in structure_list]
	return structures
