"""
The :class:`Structure` object is a collection of atoms in a periodic box.
The mandatory inputs are the cell vectors of the box and the chemical species
and *Cartesian coordinates* of the atoms.
The atoms are automatically folded back into the primary cell, so the
input coordinates don't need to lie inside the box.
"""
import numpy as np
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
                 species_labels: List[str] = None,
                 forces=None, stds=None, energy: float=None):

        # Define cell (each row is a Bravais lattice vector).
        self.cell = np.array(cell)

        # Compute the max cutoff compatible with a 3x3x3 supercell of the
        # structure.
        self.max_cutoff = get_max_cutoff(self.cell)

        # Set positions.
        self.positions = np.array(positions)

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
            assert len(positions) == len(prev_positions), \
                'Previous positions and positions are not same length'
            self.prev_positions = prev_positions

        # Set forces, energies, and stresses and their uncertainties.
        if forces is not None:
            self.forces = np.array(forces)
        else:
            self.forces = np.zeros((len(positions), 3))

        if stds is not None:
            self.stds = np.array(stds)
        else:
            self.stds = np.zeros((len(positions), 3))

        self.energy = energy

        self.local_energies = None
        self.local_energy_stds = None
        self.partial_stresses = None
        self.partial_stress_stds = None
        self.stress = None
        self.stress_stds = None
        self.potential_energy = None # duplicated with self.energy?

        self.mass_dict = mass_dict

        # Convert from elements to atomic numbers in mass dict
        if mass_dict is not None:
            keys = list(mass_dict.keys())
            for elt in keys:
                if isinstance(elt, str):
                    mass_dict[element_to_Z(elt)] = mass_dict[elt]
                    if elt.isnumeric():
                        mass_dict[int(elt)] = mass_dict[elt]

    @property
    def positions(self):
        return self._positions

    @property
    def wrapped_positions(self):
        return self._wrapped_positions

    @positions.setter
    def positions(self, position_array):
        self._positions = position_array
        self._wrapped_positions = self.wrap_positions()

    @property
    def cell(self):
        return self._cell

    @property
    def vec1(self):
        return self._vec1

    @property
    def vec2(self):
        return self._vec2

    @property
    def vec3(self):
        return self._vec3

    @property
    def cell_transpose(self):
        return self._cell_transpose

    @property
    def cell_transpose_inverse(self):
        return self._cell_transpose_inverse

    @property
    def cell_dot(self):
        return self._cell_dot

    @property
    def cell_dot_inverse(self):
        return self._cell_dot_inverse

    @cell.setter
    def cell(self, cell_array):
        """Set the cell and related properties."""
        self._cell = cell_array
        self._vec1 = cell_array[0, :]
        self._vec2 = cell_array[1, :]
        self._vec3 = cell_array[2, :]
        self._cell_transpose = cell_array.transpose()
        self._cell_transpose_inverse = np.linalg.inv(self._cell_transpose)
        self._cell_dot = self.get_cell_dot(cell_array)
        self._cell_dot_inverse = np.linalg.inv(self._cell_dot)

    @staticmethod
    def get_cell_dot(cell_array):
        """
        Compute 3x3 array of dot products of cell vectors used to
        fold atoms back to the unit cell.

        :return: 3x3 array of cell vector dot products.
        :rtype: np.ndarray
        """

        cell_dot = np.zeros((3, 3))

        for m in range(3):
            for n in range(3):
                cell_dot[m, n] = np.dot(cell_array[m], cell_array[n])

        return cell_dot

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

    def wrap_positions(self) -> 'ndarray':
        """
        Convenience function which folds atoms outside of the unit cell back
        into the unit cell. in_place flag controls if the wrapped positions
        are set in the class.

        :return: Cartesian coordinates of positions all in unit cell
        :rtype: np.ndarray
        """
        rel_pos = \
            self.raw_to_relative(self.positions, self.cell_transpose,
                                 self.cell_dot_inverse)

        rel_wrap = rel_pos - np.floor(rel_pos)

        pos_wrap = self.relative_to_raw(rel_wrap, self.cell_transpose_inverse,
                                        self.cell_dot)

        return pos_wrap

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
        struc = Structure(cell=np.array(dictionary.get('_cell',
                                                       dictionary.get(
                                                           'cell'))),
                          positions=np.array(dictionary.get('_positions',
                                                       dictionary.get(
                                                           'positions'))),
                          species=dictionary['coded_species'],
                          forces=np.array(dictionary.get('forces')),
                          mass_dict=dictionary.get('mass_dict'),
                          species_labels=dictionary.get('species_labels'),
                          energy=dictionary.get('energy', None))

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

        try:
            forces = atoms.get_forces()
        except:
            forces = None 

        try:
            stds = atoms.get_uncertainties()
        except:
            stds = None

        try:
            energy = atoms.get_potential_energy()
        except:
            energy = None

        try:
            stress = atoms.get_stress()
        except:
            stress = None 

        struc = Structure(
            cell = cell, 
            positions = atoms.positions,
            species = atoms.get_chemical_symbols(),
            forces = forces,
            stds = stds,
            energy = energy,
        )
        struc.stress = stress
        return struc

    def to_ase_atoms(self) -> 'ase.Atoms':
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms =  Atoms(self.species_labels, positions=self.positions,
                       cell=self.cell, pbc=True)

        results = {}
        properties = ["forces", "energy", "stress"]
        for p in properties:
            results[p] = getattr(self, p)
        calculator = SinglePointCalculator(atoms, **results)
        atoms.set_calculator(calculator)

        return atoms

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

    def is_valid(self, tolerance: float = .5)->bool:
        """
        Plugin to pymatgen's is_valid method to gauge if a structure
        has atoms packed too closely together, which is likely
        to cause unphysically large forces / energies or present convergence
        issues in DFT.
        :return:
        """
        pmg_structure = self.to_pmg_structure()
        return pmg_structure.is_valid(tol=tolerance)


    def to_xyz(self, extended_xyz: bool = True, print_stds: bool = False,
               print_forces: bool = False, print_max_stds: bool = False,
               print_energies: bool = False, predict_energy = None,
               dft_forces = None, dft_energy = None, timestep=-1,
               write_file: str = '', append: bool = False) -> str:
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
            if timestep > 0:
                xyz_str += f' Timestep={timestep}'
            if predict_energy:
                xyz_str += f' PE={predict_energy}'
            if dft_energy is not None:
                xyz_str += f' DFT_PE={dft_energy}'
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
            if print_energies:
                if self.local_energies is None:
                    print_energies = False
                else:
                    xyz_str += ':local_energy:R:1'
                    local_energies = self.local_energies
            if dft_forces is not None:
                xyz_str += ':dft_forces:R:3'
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
            if print_energies and extended_xyz:
                xyz_str += f" {local_energies[i]}"
            if print_max_stds and extended_xyz:
                xyz_str += f" {np.max(stds[i,:])} "
            if dft_forces is not None:
                xyz_str += f' {dft_forces[i, 0]} {dft_forces[i,1]} ' \
                          f'{dft_forces[i, 2]}'
            if i < (len(self.positions)-1):
                xyz_str += '\n'

        # Write to file, optionally
        if write_file:
            if append:
                fmt = 'a'
            else:
                fmt = 'w'
            with open(write_file, fmt) as f:
                f.write(xyz_str)
                f.write("\n")

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
    coded_species = []
    for spec in species:
        if spec in unique_species:
            coded_species.append(unique_species.index(spec))
        else:
            coded_species.append(len(unique_species))
            unique_species.append(spec)
    coded_species = np.array(coded_species)

    return unique_species, coded_species
