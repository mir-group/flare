import numpy as np
from typing import List
from flare.util import element_to_Z


class Structure(object):
    """
        Contains positions, species, cell, cutoff rad, previous positions,
        forces, and stds of forces, computes inv_cell and bond list

        :param cell: nparray, 3x3 Bravais cell
        :param species: list[int], List of integers corresponding to elements
        :param positions: list[nparray] list of positions
        :param cutoff: float, Cutoff radius for GP

    """

    def __init__(self, cell: np.ndarray, species: List[int],
                 positions: np.ndarray, mass_dict: dict = None,
                 prev_positions: np.ndarray = None, species_labels=None):
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

        # get unique species
        # self.species = species

        # If species are strings, convert species to integers by atomic number
        species = [element_to_Z(spec) for spec in species]

        self.coded_species = np.array(species)
        self.species_labels = species_labels
        self.nat = len(species)
        # unique_species, coded_species = self.get_unique_species(species)
        # self.unique_species = unique_species
        # self.coded_species = coded_species

        # Default: atoms have no velocity
        if prev_positions is None:
            self.prev_positions = np.copy(self.positions)
        else:
            assert len(positions) == len(prev_positions), 'Previous ' \
                                                          'positions and ' \
                                                          'positions are not same length'
            self.prev_positions = prev_positions

        self.energy = None
        self.stress = None
        self.forces = np.zeros((len(positions), 3))
        self.stds = np.zeros((len(positions), 3))
        self.mass_dict = mass_dict

    def get_cell_dot(self):
        cell_dot = np.zeros((3, 3))

        for m in range(3):
            for n in range(3):
                cell_dot[m, n] = np.dot(self.cell[m], self.cell[n])

        return cell_dot

    @staticmethod
    def raw_to_relative(positions, cell_transpose, cell_dot_inverse):
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
