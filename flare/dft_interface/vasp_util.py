from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Vasprun
from subprocess import call
import numpy as np
from flare.struc import Structure
from typing import List, Union
from json import dump, load
from flare.util import NumpyEncoder
import os, shutil

def check_vasprun(vasprun):
    if type(vasprun) == str:
        return Vasprun(vasprun)
    elif type(vasprun) == Vasprun:
        return vasprun
    else:
        raise ValueError('Vasprun argument is not a string or Vasprun instance!')

def run_dft(calc_dir: str, dft_loc: str,
    structure: Structure=None, en: bool=False, vasp_cmd="{}"):
    
    currdir = os.getcwd()
    if structure:
        edit_dft_input_positions("POSCAR", structure)
    try:
        if currdir != calc_dir:
            os.chdir(calc_dir)
        call(vasp_cmd.format(dft_loc).split())

        if en:
            parse_func = parse_dft_forces_and_energy
        else:
            parse_func = parse_dft_forces

        try:
            forces = parse_func("vasprun.xml")
        except FileNotFoundError:
            raise FileNotFoundError("""Could not load vasprun.xml. The calculation may not have finished.
                                        Current directory is %s""" % os.getcwd())

        os.chdir(currdir)
        return forces

    except Exception as e:
        os.chdir(currdir)
        raise e
    

def dft_input_to_structure(poscar: str):
    """
    Parse the DFT input in a directory.
    :param vasp_input: directory of vasp input
    """
    return Structure.from_pmg_structure(Poscar.from_file(poscar).structure)


def edit_dft_input_positions(poscar: str, structure: Structure):
    """
    WARNING: Destructively replaces the file with the name specified by POSCAR
    """
    if os.path.isfile(poscar):
        shutil.copyfile(poscar, poscar+'.bak')
    f = open(poscar, 'w')
    f.write(Poscar(structure.to_pmg_structure()).get_string(significant_figures=15))
    f.close()
    return poscar


def parse_dft_forces(vasprun: Union[str, Vasprun]):
    vasprun = check_vasprun(vasprun)
    istep = vasprun.ionic_steps[-1]
    return np.array(istep['forces'])


def parse_dft_forces_and_energy(vasprun: Union[str, Vasprun]):
    vasprun = check_vasprun(vasprun)
    istep = vasprun.ionic_steps[-1]
    return np.array(istep.forces), istep["electronic_steps"][-1]["e_0_energy"]


def md_trajectory_from_vasprun(vasprun: Union[str, Vasprun], ionic_step_skips=1):
    vasprun = check_vasprun(vasprun)

    struc_lst = []
    for step in vasprun.ionic_steps[::ionic_step_skips]:
        structure = Structure.from_pmg_structure(step['structure'])
        structure.energy = step["electronic_steps"][-1]["e_0_energy"]
        #TODO should choose e_wo_entrp or e_fr_energy?
        structure.forces = np.array(step["forces"])
        structure.stress = np.array(step["stress"])
        struc_lst.append(structure)

    return struc_lst
