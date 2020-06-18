import numpy as np
import os
import shutil

from json import dump, load
from subprocess import call
from typing import List, Union

from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.core.periodic_table import Element

from flare.struc import Structure
from flare.utils.element_coder import NumpyEncoder

name="VASP"

def check_vasprun(vasprun: Union[str, Vasprun], vasprun_kwargs: dict = {}) -> \
        Vasprun:
    """
    Helper utility to take a vasprun file name or Vasprun object
    and return a vasprun object.
    :param vasprun: vasprun filename or object
    """

    if type(vasprun) == str:
        return Vasprun(vasprun, **vasprun_kwargs)
    elif type(vasprun) == Vasprun:
        return vasprun
    else:
        raise ValueError('Vasprun argument is not a string or '
                         'Vasprun instance!')


def parse_dft_input(input: str):
    """
    Returns the positions, species, and cell of a POSCAR file.
    Outputs are specced for OTF module.

    :param input: POSCAR file input
    :return:
    """

    pmg_structure = Poscar.from_file(input).structure
    flare_structure = Structure.from_pmg_structure(pmg_structure)

    positions = flare_structure.positions
    species = flare_structure.species_labels
    cell = flare_structure.cell
    #TODO Allow for custom masses in POSCAR

    elements = set(species)

    # conversion from amu to md units
    mass_dict = {elt: Element(elt).atomic_mass * 0.000103642695727 for
                 elt in elements}

    return positions, species, cell, mass_dict

def run_dft(calc_dir: str, dft_loc: str,
            structure: Structure = None, en: bool = False, vasp_cmd="{}"):
    """
    Run a VASP calculation.
    :param calc_dir: Name of directory to perform calculation in
    :param dft_loc: Name of VASP command (i.e. executable location)
    :param structure: flare structure object
    :param en: whether to return the final energy along with the forces
    :param vasp_cmd: Command to run VASP (leave a formatter "{}" to insert dft_loc);
        this can be used to specify mpirun, srun, etc.
    Returns:
        forces on each atom (and energy if en=True)
    """

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
            os.chdir(currdir)
            raise FileNotFoundError("Could not load vasprun.xml."\
                                    "The calculation may not have finished."
                                    f"Current directory is {os.getcwd()}")

        os.chdir(currdir)
        return forces

    except Exception as e:
        os.chdir(currdir)
        raise e


def run_dft_par(dft_input: str, structure: Structure,
                dft_command:str= None, n_cpus=1,
                dft_out="vasprun.xml",
                parallel_prefix="mpi",
                mpi = None, npool = None,
                screen_out='vasp.out',
                **dft_kwargs):
    # TODO Incorporate Custodian.
    edit_dft_input_positions(dft_input, structure)

    if dft_command is None or not os.environ.get('VASP_COMMAND'):
        raise FileNotFoundError\
            ("Warning: No VASP Command passed, or stored in "
            "environment as VASP_COMMAND. ")

    if n_cpus > 1:
        # why is parallel prefix needed?
        if (parallel_prefix == "mpi") and (mpi == 'mpi'):
            dft_command = f'mpirun -np {n_cpus} {dft_command}'
        elif mpi == 'srun':
            dft_command = f'srun -n {n_cpus} {dft_command}'

    else:
        serial_prefix = dft_kwargs.get('serial_prefix', '')
        dft_command = f'{serial_prefix} {dft_command}'

    with open(screen_out, "w+") as fout:
        call(dft_command.split(), stdout=fout)

    return parse_dft_forces(dft_out)


def dft_input_to_structure(poscar: str):
    """
    Parse the DFT input in a directory.
    :param vasp_input: directory of vasp input
    """
    return Structure.from_pmg_structure(Poscar.from_file(poscar).structure)


def edit_dft_input_positions(output_name: str, structure: Structure):
    """
    Writes a VASP POSCAR file from structure with the name poscar .
    WARNING: Destructively replaces the file with the name specified by poscar
    :param poscar: Name of file
    :param structure: structure to write to file
    """
    if os.path.isfile(output_name):
        shutil.copyfile(output_name, output_name + '.bak')
    poscar = Poscar(structure.to_pmg_structure())
    poscar.write_file(output_name)
    return output_name


def parse_dft_forces(vasprun: Union[str, Vasprun]):
    """
    Parses the DFT forces from the last ionic step of a VASP vasprun.xml file
    :param vasprun: pymatgen Vasprun object or vasprun filename
    """
    vasprun = check_vasprun(vasprun)
    istep = vasprun.ionic_steps[-1]
    return np.array(istep['forces'])


def parse_dft_forces_and_energy(vasprun: Union[str, Vasprun]):
    """
    Parses the DFT forces and energy from a VASP vasprun.xml file
    :param vasprun: pymatgen Vasprun object or vasprun filename
    """
    vasprun = check_vasprun(vasprun)
    istep = vasprun.ionic_steps[-1]
    return np.array(istep['forces']), \
           istep["electronic_steps"][-1]["e_0_energy"]


def md_trajectory_from_vasprun(vasprun: Union[str, Vasprun],
                               ionic_step_skips=1, vasprun_kwargs: dict={}):
    """
    Returns a list of flare Structure objects decorated with forces, stress,
    and total energy from a MD trajectory performed in VASP.
    :param vasprun: pymatgen Vasprun object or vasprun filename
    :param ionic_step_skips: if True, only samples the configuration every
        ionic_skip_steps steps.
    """
    vasprun = check_vasprun(vasprun, vasprun_kwargs)

    struc_lst = []
    for step in vasprun.ionic_steps[::ionic_step_skips]:
        structure = Structure.from_pmg_structure(step["structure"])
        structure.energy = step["electronic_steps"][-1]["e_0_energy"]
        # TODO should choose e_wo_entrp or e_fr_energy?
        structure.forces = np.array(step.get("forces"))
        structure.stress = np.array(step.get("stress"))
        struc_lst.append(structure)

    return struc_lst
