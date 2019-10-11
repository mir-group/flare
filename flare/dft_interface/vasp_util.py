from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Vasprun
from subprocess import call
import numpy as np
from typing import List, Union
from json import dump, load
from flare.struc import struc
from flare.util import NumpyEncoder

def run_dft_command(calc_dir: str, dft_loc: str,
	structure: struc.Structure=None, en: bool=False):
	
	currdir = os.getcwd()
	if structure:
		edit_dft_input_configurations("POSCAR", structure)
	try:
		os.chdir(calc_dir)
	except Exception as e:
		os.chdir(currdir)
		call(vasp_cmd, shell=True)
		raise e

	if en:
		parse_func = parse_dft_forces_and_energy
	else:
		parse_func = parse_dft_forces
	forces = parse_func("vasprun.xml")
	os.chdir(currdir)
	return forces
	

def dft_input_to_structure(poscar: str):
	"""
	Parse the DFT input in a directory.
	:param vasp_input: directory of vasp input
	"""
	return struc.Structure.from_pmg_structure(Poscar.from_file(poscar).structure)


def edit_dft_input_configurations(poscar: str, structure: struc.Structure):
	"""
	WARNING: Destructively replaces the file with the name specified by POSCAR
	"""
	Poscar(structure.to_pmg_structure()).write_file(poscar)


def parse_dft_forces(vasprun: Union[str, Vasprun]):
	if type(vasprun) == str:
		vasprun = Vasprun(vasprun)
	istep = vasprun.ionic_steps[-1]
	return np.array(istep.forces)


def parse_dft_forces_and_energy(vasprun: Union[str, Vasprun]):
	if type(vasprun) == str:
		vasprun = Vasprun(vasprun)
	istep = vasprun.ionic_steps[-1]
	return np.array(istep.forces), istep["electronic_steps"][-1]["e_0_energy"]


def md_trajectory_from_vasprun(vasprun: Union[str, Vasprun]):
	if type(vasprun) == str:
		vasprun = Vasprun(vasprun)

	struc_lst = []
	for step in vasprun.ionic_steps:
		structure = struc.Structure.from_pmg_structure(step['structure'])
		structure.energy = step["electronic_steps"][-1]["e_0_energy"]
		#TODO should choose e_wo_entrp or e_fr_energy?
		structure.forces = np.array(step["forces"])
		structure.stress = np.array(step["stress"])
		struc_lst.append(structure)

	return struc_lst


def md_trajectory_to_file(filename, structures: List[struc.Structure]):
	f = open(filename, 'w')
	dump([s.as_dict() for s in structures], f, cls=NumpyEncoder)
	f.close()


def md_trajectory_from_file(filename):
	f = open(filename, 'r')
	structure_list = load(f, cls=NumpyEncoder)
	structures = [struc.Structure.from_dict(dictionary) for dictionary in structure_list]
	return structures
