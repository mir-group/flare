"""
The :class:`Structure` object is a collection of atoms in a periodic box.
The mandatory inputs are the cell vectors of the box and the chemical species
and *Cartesian coordinates* of the atoms.
The atoms are automatically folded back into the primary cell, so the
input coordinates don't need to lie inside the box.
Energy, force, and stress information can be included which can then be
used to train ML models.
"""
import numpy as np
from flare.util import element_to_Z, Z_to_element, NumpyEncoder
from json import dumps, loads

from typing import List, Union, Any

try:
    # Used for to_pmg_structure method
    import pymatgen.core.structure as pmgstruc
    import pymatgen.io.vasp.inputs as pmgvaspio
    _pmg_present = True
except ImportError:
    _pmg_present = False


class Structure:
    """
    Contains information about a periodic structure of atoms, including the
    periodic cell boundaries, atomic species, and coordinates.

    *Note that input positions are assumed to be Cartesian.*

    :param cell: 3x3 array whose rows are the Bravais lattice vectors of the
        cell.
    :type cell: np.ndarray
    :param species: List of atomic species, which are represented either as
        integers or chemical symbols.
    :type species: List
    :param positions: Nx3 array of atomic coordinates.
    :type positions: np.ndarray
    :param mass_dict: Dictionary of atomic masses used in MD simulations.
    :type mass_dict: dict
    :param prev_positions: Nx3 array of previous atomic coordinates used in
        MD simulations.
    :type prev_positions: np.ndarray
    :param species_labels: List of chemical symbols. Used in the output file
        of on-the-fly runs.
    :type species_labels: List[str]
    :param stds: Uncertainty associated with forces
    :type stds: np.ndarray
    """

    def __init__(self, cell: 'ndarray', species: Union[List[str], List[int]],
                 positions: 'ndarray', mass_dict: dict = None,
                 prev_positions: 'ndarray' = None,
                 species_labels: List[str] = None, energy=None,
                 forces=None, stress=None, stds=None):
        # Set up individual Bravais lattice vectors
        self.cell = np.array(cell)
        self.vec1 = self.cell[0, :]
        self.vec2 = self.cell[1, :]
        self.vec3 = self.cell[2, :]

        # Compute the max cutoff for sweep = 1.
        self.max_cutoff = self.get_max_cutoff()

        # get cell matrices for wrapping coordinates
        self.cell_transpose = self.cell.transpose()
        self.cell_transpose_inverse = np.linalg.inv(self.cell_transpose)
        self.cell_dot = self.get_cell_dot()
        self.cell_dot_inverse = np.linalg.inv(self.cell_dot)

        # set positions
        self.positions = np.array(positions)
        self.wrapped_positions = self.wrap_positions(in_place=False)

        # If species are strings, convert species to integers by atomic number
        if species_labels is None:
            self.species_labels = species
        else:
            self.species_labels = species_labels
        self.coded_species = np.array([element_to_Z(spec) for spec in species])
        self.nat = len(species)

        # Default: atoms have no velocity
        if prev_positions is None:
            self.prev_positions = np.copy(self.positions)
        else:
            assert len(positions) == len(prev_positions), 'Previous ' \
                                                          'positions and ' \
                                                          'positions are not' \
                                                          'same length'
            self.prev_positions = prev_positions

        # assign structure labels
        self.energy = energy
        self.stress = stress
        self.forces = forces
        self.labels = self.get_labels()

        if stds is not None:
            self.stds = np.array(stds)
        else:
            self.stds = np.zeros((len(positions), 3))

        self.mass_dict = mass_dict

        # Convert from elements to atomic numbers in mass dict
        if mass_dict is not None:
            keys = list(mass_dict.keys())
            for elt in keys:
                if isinstance(elt, str):
                    mass_dict[element_to_Z(elt)] = mass_dict[elt]
                    if elt.isnumeric():
                        mass_dict[int(elt)] = mass_dict[elt]

    def get_cell_dot(self):
        """
        Compute 3x3 array of dot products of cell vectors used to
        fold atoms back to the unit cell.

        :return: 3x3 array of cell vector dot products.
        :rtype: np.ndarray
        """

        cell_dot = np.zeros((3, 3))

        for m in range(3):
            for n in range(3):
                cell_dot[m, n] = np.dot(self.cell[m], self.cell[n])

        return cell_dot

    def get_max_cutoff(self) -> float:
        """Compute the maximum cutoff compatible with a 3x3x3 supercell of the
            structure.

        Returns:
            float: maximum cutoff
        """
        # Retrieve the lattice vectors.
        a_vec = self.cell[0]
        b_vec = self.cell[1]
        c_vec = self.cell[2]

        # Compute dot products and norms of lattice vectors.
        a_dot_b = np.dot(a_vec, b_vec)
        a_dot_c = np.dot(a_vec, c_vec)
        b_dot_c = np.dot(b_vec, c_vec)

        a_norm = np.linalg.norm(a_vec)
        b_norm = np.linalg.norm(b_vec)
        c_norm = np.linalg.norm(c_vec)

        # Compute the six independent altitudes of the cell faces.
        # The smallest is the maximum atomic environment cutoff that can be
        # used with sweep=1.
        max_candidates = np.zeros(6)
        max_candidates[0] = \
            a_norm * np.sqrt(1 - (a_dot_b / (a_norm * b_norm))**2)
        max_candidates[1] = \
            b_norm * np.sqrt(1 - (a_dot_b / (a_norm * b_norm))**2)
        max_candidates[2] = \
            a_norm * np.sqrt(1 - (a_dot_c / (a_norm * c_norm))**2)
        max_candidates[3] = \
            c_norm * np.sqrt(1 - (a_dot_c / (a_norm * c_norm))**2)
        max_candidates[4] = \
            b_norm * np.sqrt(1 - (b_dot_c / (b_norm * c_norm))**2)
        max_candidates[5] = \
            c_norm * np.sqrt(1 - (b_dot_c / (b_norm * c_norm))**2)

        return np.min(max_candidates)

    @staticmethod
    def raw_to_relative(positions: 'ndarray', cell_transpose: 'ndarray',
                        cell_dot_inverse: 'ndarray') -> 'ndarray':
        """Convert Cartesian coordinates to relative (fractional) coordinates,
        expressed in terms of the cell vectors set in self.cell.

        :param positions: Cartesian coordinates.
        :type positions: np.ndarray
        :param cell_transpose: Transpose of the cell array.
        :type cell_transpose: np.ndarray
        :param cell_dot_inverse: Inverse of the array of dot products of
            cell vectors.
        :type cell_dot_inverse: np.ndarray
        :return: Relative positions.
        :rtype: np.ndarray
        """

        relative_positions = \
            np.matmul(np.matmul(positions, cell_transpose),
                      cell_dot_inverse)

        return relative_positions

    @staticmethod
    def relative_to_raw(relative_positions: 'ndarray',
                        cell_transpose_inverse: 'ndarray',
                        cell_dot: 'ndarray') -> 'ndarray':
        """Convert fractional coordinates to raw (Cartesian) coordinates.

        :param relative_positions: fractional coordinates.
        :type relative_positions: np.ndarray
        :param cell_transpose_inverse: Transpose of the cell array.
        :type cell_transpose_inverse: np.ndarray
        :param cell_dot: Dot products of cell vectors
        :type cell_dot: np.ndarray
        :return: Cartesian positions.
        :rtype: np.ndarray
        """

        return np.matmul(np.matmul(relative_positions, cell_dot),
                         cell_transpose_inverse)

    def wrap_positions(self, in_place: bool = True) -> 'ndarray':
        """
        Convenience function which folds atoms outside of the unit cell back
        into the unit cell. in_place flag controls if the wrapped positions
        are set in the class.

        :param in_place: If true, set the current structure positions to be
            the wrapped positions.
        :return: Cartesian coordinates of positions all in unit cell
        :rtype: np.ndarray
        """
        rel_pos = \
            self.raw_to_relative(self.positions, self.cell_transpose,
                                 self.cell_dot_inverse)

        rel_wrap = rel_pos - np.floor(rel_pos)

        pos_wrap = self.relative_to_raw(rel_wrap, self.cell_transpose_inverse,
                                        self.cell_dot)

        if in_place:
            self.wrapped_positions = pos_wrap

        return pos_wrap

    def get_labels(self):
        labels = []
        if self.energy is not None:
            labels.append(self.energy)
        if self.forces is not None:
            unrolled_forces = self.forces.reshape(-1)
            for force_comp in unrolled_forces:
                labels.append(force_comp)
        if self.stress is not None:
            labels.extend([self.stress[0, 0], self.stress[0, 1],
                           self.stress[0, 2], self.stress[1, 1],
                           self.stress[1, 2], self.stress[2, 2]])

        labels = np.array(labels)

        return labels

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
        return dict(vars(self))

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
        struc = Structure(cell=np.array(dictionary['cell']),
                          positions=np.array(dictionary['positions']),
                          species=dictionary['coded_species'],
                          forces=np.array(dictionary.get('forces')),
                          mass_dict=dictionary.get('mass_dict'),
                          species_labels=dictionary.get('species_labels'))

        struc.energy = dictionary.get('energy')
        struc.stress = dictionary.get('stress')
        struc.stds = np.array(dictionary.get('stds'))

        return struc

    @staticmethod
    def from_ase_atoms(atoms: 'ase.Atoms') -> 'flare.struc.Structure':
        """
        From an ASE Atoms object, return a FLARE structure

        :param atoms: ASE Atoms object
        :type atoms: ASE Atoms object
        :return: A FLARE structure from an ASE atoms object
        """
        struc = Structure(cell=np.array(atoms.cell),
                          positions=atoms.positions,
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

        return pmgstruc.Structure(lattice=self.cell,
                                  species=self.species_labels,
                                  coords=self.positions,
                                  coords_are_cartesian=True,
                                  site_properties=site_properties
                                  )

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
                                                 List['flare.struc.Structure']
                                                 ]:
        """
        Load a FLARE structure from a file or a series of FLARE structures
        :param file_name:
        :param format:
        :return:
        """

        try:
            with open(file_name, 'r') as _:
                pass
        except FileNotFoundError:
            raise FileNotFoundError

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
    coded_species = []
    for spec in species:
        if spec in unique_species:
            coded_species.append(unique_species.index(spec))
        else:
            coded_species.append(len(unique_species))
            unique_species.append(spec)
    coded_species = np.array(coded_species)

    return unique_species, coded_species
