import numpy as np
from typing import List
from flare.util import element_to_Z, NumpyEncoder
from json import dumps


class Structure(object):
    """
    Contains information about a structure of atoms, including the periodic
    cell boundaries and atomic species and coordinates.
    
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
    """

    def __init__(self, cell, species, positions, mass_dict=None,
                 prev_positions=None, species_labels=None, energy=None,
                 forces=None, stress=None):
        self.cell = cell
        self.vec1 = cell[0, :]
        self.vec2 = cell[1, :]
        self.vec3 = cell[2, :]

        # get cell matrices for wrapping coordinates
        self.cell_transpose = self.cell.transpose()
        self.cell_transpose_inverse = np.linalg.inv(self.cell_transpose)
        self.cell_dot = self.get_cell_dot()
        self.cell_dot_inverse = np.linalg.inv(self.cell_dot)

        # set positions
        self.positions = np.array(positions)
        self.wrap_positions()

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

        self.energy = energy
        self.forces = forces
        self.stress = stress
        self.labels = self.get_labels()

        self.stds = np.zeros((len(positions), 3))
        self.mass_dict = mass_dict

    def get_cell_dot(self):
        """
        Compute 3x3 array of dot products of cell vectors used to fold atoms
        back to the unit cell.

        :return: 3x3 array of cell vector dot products.
        :rtype: np.ndarray
        """

        cell_dot = np.zeros((3, 3))

        for m in range(3):
            for n in range(3):
                cell_dot[m, n] = np.dot(self.cell[m], self.cell[n])

        return cell_dot

    @staticmethod
    def raw_to_relative(positions, cell_transpose, cell_dot_inverse):
        """Convert Cartesian coordinates to relative coordinates expressed in
        terms of the cell vectors.
        
        :param positions: Cartesian coordinates.
        :type positions: np.ndarray
        :param cell_transpose: Transpose of the cell array.
        :type cell_transpose: np.ndarray
        :param cell_dot_inverse: Inverse of the array of dot products of cell
        vectors.
        :type cell_dot_inverse: np.ndarray
        :return: Relative positions.
        :rtype: np.ndarray
        """

        relative_positions = \
            np.matmul(np.matmul(positions, cell_transpose),
                      cell_dot_inverse)

        return relative_positions

    @staticmethod
    def relative_to_raw(relative_positions, cell_transpose_inverse,
                        cell_dot):
        positions = \
            np.matmul(np.matmul(relative_positions, cell_dot),
                      cell_transpose_inverse)

        return positions

    def wrap_positions(self):
        rel_pos = \
            self.raw_to_relative(self.positions, self.cell_transpose,
                                 self.cell_dot_inverse)

        rel_wrap = rel_pos - np.floor(rel_pos)

        pos_wrap = self.relative_to_raw(rel_wrap, self.cell_transpose_inverse,
                                        self.cell_dot)

        self.wrapped_positions = pos_wrap

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

    def indices_of_specie(self, specie: int):
        """
        Return the indicies of atoms in  the structure which are atoms
        corresponding to a given specie
        :param specie:
        :return:
        """
        return [i for i, spec in enumerate(self.coded_species)
                if spec == specie]

    # TODO make more descriptive
    def __str__(self):
        return 'Structure with {} atoms of types {}'.format(self.nat,
                                                     set(self.species_labels))

    def __len__(self):
        return self.nat

    def as_dict(self):
        """
        Returns structure as a dictionary for serialization
        purposes.
        :return:
        """
        return dict(vars(self))

    def as_str(self):
        return dumps(self.as_dict(), cls=NumpyEncoder)

    @staticmethod
    def from_dict(dictionary):
        struc = Structure(cell=np.array(dictionary['cell']),
                          positions=np.array(dictionary['positions']),
                          species=dictionary['coded_species'])

        struc.forces = np.array(dictionary['forces'])
        struc.stress = dictionary['stress']
        struc.stds = np.array(dictionary['stds'])
        struc.mass_dict = dictionary['mass_dict']
        struc.species_labels = dictionary['species_labels']

        return struc


def get_unique_species(species):
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


if __name__ == '__main__':
    pass
