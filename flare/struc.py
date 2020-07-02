"""
The :class:`Structure` object is a collection of atoms in a periodic box.
The mandatory inputs are the cell vectors of the box and the chemical species
and *Cartesian coordinates* of the atoms.
The atoms are automatically folded back into the primary cell, so the
input coordinates don't need to lie inside the box.
"""
import numpy as np
from flare._C_flare import Structure
from flare.utils.element_coder import element_to_Z, Z_to_element, NumpyEncoder
from flare.utils.learner import get_max_cutoff
from json import dumps, loads

from typing import List, Union, Any

try:
    # Used for to_pmg_structure method
    import pymatgen.core.structure as pmgstruc
    import pymatgen.io.vasp.inputs as pmgvaspio
    _pmg_present = True
except ImportError:
    _pmg_present = False


def indices_of_specie(self, specie: Union[int, str]) -> List[int]:
    """
    Return the indices of a given species within atoms of the structure.

    :param specie: Element to target, can be string or integer
    :return: The indices in the structure at which this element occurs
    :rtype: List[str]
    """
    return [i for i, spec in enumerate(self.coded_species)
            if spec == specie]


# TODO make more descriptive
def __str__(self) -> str:
    """
    Simple descriptive string of structure.

    :return: One-line descriptor of number of atoms and species present.
    :rtype: str
    """

    return 'Structure with {} atoms of types {}'\
        .format(self.nat, set(self.species_labels))


def __len__(self) -> int:
    """
    Returns number of atoms in structure.

    :return: number of atoms in structure.
    :rtype: int
    """
    return self.nat


def as_dict(self) -> dict:
    """
    Returns structure as a dictionary; useful for serialization purposes.

    :return: Dictionary version of current structure
    :rtype: dict
    """
    return {'positions': self.positions, 'cell': self.cell,
            'species': self.coded_species, 'mass_dict': self.mass_dict,
            'prev_positions': self.prev_positions,
            'species_labels': self.species_labels,
            'forces': self.forces, 'stds': self.stds}


def as_str(self) -> str:
    """
    Returns string dictionary serialization cast as string.

    :return: output of as_dict method cast as string
    :rtype: str
    """
    return dumps(self.as_dict(), cls=NumpyEncoder)


@staticmethod
def from_dict(dictionary: dict) -> 'flare.struc.Structure':
    """
    Assembles a Structure object from a dictionary parameterizing one.

    :param dictionary: dict describing structure parameters.
    :return: FLARE structure assembled from dictionary
    """
    struc = Structure(
        cell=np.array(dictionary['cell']),
        positions=np.array(dictionary['positions']),
        species=dictionary['species'],
        mass_dict=dictionary.get('mass_dict'),
        species_labels=dictionary.get('species_labels'))

    struc.forces = np.array(dictionary.get('forces'))
    struc.stds = np.array(dictionary.get('stds'))

    return struc


@staticmethod
def from_ase_atoms(atoms: 'ase.Atoms', cell=None) -> 'flare.struc.Structure':
    """
    From an ASE Atoms object, return a FLARE structure

    :param atoms: ASE Atoms object
    :type atoms: ASE Atoms object
    :return: A FLARE structure from an ASE atoms object
    """

    if cell is None:
        cell = np.array(atoms.cell)
    struc = Structure(cell=cell, positions=atoms.positions,
                      species=atoms.get_chemical_symbols())
    return struc


def to_ase_atoms(self) -> 'ase.Atoms':
    from ase import Atoms
    return Atoms(self.species_labels, positions=self.positions,
                 cell=self.cell, pbc=True)


def to_pmg_structure(self):
    """
    Returns FLARE structure as a pymatgen structure.

    :return: Pymatgen structure corresponding to current FLARE structure
    """

    if not _pmg_present:
        raise ModuleNotFoundError("Pymatgen is not present. Please "
                                    "install Pymatgen and try again")

    if self.forces is None:
        forces_temp = np.zeros((len(self.positions), 3))
        site_properties = {'force:': forces_temp, 'std': self.stds}
    else:
        site_properties = {'force:': self.forces, 'std': self.stds}

    return pmgstruc.Structure(
        lattice=self.cell, species=self.species_labels,
        coords=self.positions, coords_are_cartesian=True,
        site_properties=site_properties)


@staticmethod
def from_pmg_structure(structure: 'pymatgen Structure') -> \
        'flare Structure':
    """
    Returns Pymatgen structure as FLARE structure.

    :param structure: Pymatgen Structure
    :type structure: Pymatgen Structure
    :return: FLARE Structure
    """

    cell = structure.lattice.matrix.copy()
    species = [str(spec) for spec in structure.species]
    positions = structure.cart_coords.copy()

    new_struc = Structure(cell=cell, species=species,
                          positions=positions)

    site_props = structure.site_properties

    if 'force' in site_props.keys():
        forces = site_props['force']
        new_struc.forces = [np.array(force) for force in forces]

    if 'std' in site_props.keys():
        stds = site_props['std']
        new_struc.stds = [np.array(std) for std in stds]

    return new_struc


def to_xyz(self, extended_xyz: bool = True, print_stds: bool = False,
           print_forces: bool = False, print_max_stds: bool = False,
           write_file: str = '') -> str:
    """
    Convenience function which turns a structure into an extended .xyz
    file; useful for further input into visualization programs like VESTA
    or Ovito. Can be saved to an output file via write_file.

    :param print_stds: Print the stds associated with the structure.
    :param print_forces:
    :param extended_xyz:
    :param print_max_stds:
    :param write_file:
    :return:
    """
    species_list = [Z_to_element(x) for x in self.coded_species]
    xyz_str = ''
    xyz_str += f'{len(self.coded_species)} \n'

    # Add header line with info about lattice and properties if extended
    #  xyz option is called.
    if extended_xyz:
        cell = self.cell

        xyz_str += f'Lattice="{cell[0,0]} {cell[0,1]} {cell[0,2]}'
        xyz_str += f' {cell[1,0]} {cell[1,1]} {cell[1,2]}'
        xyz_str += f' {cell[2,0]} {cell[2,1]} {cell[2,2]}"'
        xyz_str += f' Proprties="species:S:1:pos:R:3'

        if print_stds:
            xyz_str += ':stds:R:3'
            stds = self.stds
        if print_forces:
            xyz_str += ':forces:R:3'
            forces = self.forces
        if print_max_stds:
            xyz_str += ':max_std:R:1'
            stds = self.stds
        xyz_str += '\n'
    else:
        xyz_str += '\n'

    for i, pos in enumerate(self.positions):
        # Write positions
        xyz_str += f"{species_list[i]} {pos[0]} {pos[1]} {pos[2]}"

        # If extended XYZ: Add in extra information
        if print_stds and extended_xyz:
            xyz_str += f" {stds[i,0]} {stds[i,1]} {stds[i,2]}"
        if print_forces and extended_xyz:
            xyz_str += f" {forces[i,0]} {forces[i,1]} {forces[i,2]}"
        if print_max_stds and extended_xyz:
            xyz_str += f" {np.max(stds[i,:])} "
        xyz_str += '\n'

    # Write to file, optionally
    if write_file:
        with open(write_file, 'w') as f:
            f.write(xyz_str)

    return xyz_str


@staticmethod
def from_file(file_name, format='') -> Union['flare.struc.Structure',
                                             List['flare.struc.Structure']]:
    """
    Load a FLARE structure from a file or a series of FLARE structures
    :param file_name:
    :param format:
    :return:
    """

    # Ensure the file specified exists.
    with open(file_name, 'r') as _:
        pass

    if 'xyz' in file_name or 'xyz' in format.lower():
        raise NotImplementedError

    if 'json' in format.lower() or '.json' in file_name:
        # Assumed format is one FLARE structure per line,
        # or one line with many FLARE structures
        with open(file_name, 'r') as f:
            thelines = f.readlines()

            non_empty_lines = [loads(line) for line in thelines if
                               len(line) > 2]

        structures = [Structure.from_dict(struc_dict) for struc_dict in
                      non_empty_lines]

        if len(structures) == 1:
            return structures[0]
        else:
            return structures

    is_poscar = 'POSCAR' in file_name or 'CONTCAR' in file_name \
        or 'vasp' in format.lower()
    if is_poscar and _pmg_present:
        pmg_structure = pmgvaspio.Poscar.from_file(file_name).structure
        return Structure.from_pmg_structure(pmg_structure)
    elif is_poscar and not _pmg_present:
        raise ImportError("Pymatgen not imported; "
                          "functionality requires pymatgen.")


def get_unique_species(species: List[Any]) -> (List, List[int]):
    """
    Returns a list of the unique species passed in, and a list of
    integers indexing them.

    :param species: Species to index uniquely
    :return: List of the unique species, and integer indexes
    """
    unique_species = []
    species = []
    for spec in species:
        if spec in unique_species:
            species.append(unique_species.index(spec))
        else:
            species.append(len(unique_species))
            unique_species.append(spec)
    species = np.array(species)

    return unique_species, species


# Add pure Python methods to the structure class.
Structure.indices_of_specie = indices_of_specie
Structure.__str__ = __str__
Structure.__len__ = __len__
Structure.as_dict = as_dict
Structure.as_str = as_str
Structure.from_dict = from_dict
Structure.from_ase_atoms = from_ase_atoms
Structure.to_ase_atoms = to_ase_atoms
Structure.to_pmg_structure = to_pmg_structure
Structure.from_pmg_structure = from_pmg_structure
Structure.to_xyz = to_xyz
Structure.from_file = from_file
Structure.get_unique_species = get_unique_species
