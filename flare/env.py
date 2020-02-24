"""The :class:`AtomicEnvironment` object stores information about the local
environment of an atom. :class:`AtomicEnvironment` objects are inputs to the
2-, 3-, and 2+3-body kernels."""
import numpy as np
from math import sqrt
from numba import njit
from flare.struc import Structure


class AtomicEnvironment:
    """
    Contains information about the local environment of an atom, including
    arrays of pair and triplet distances and the chemical species of atoms
    in the environment.

    :param structure: Structure of atoms.
    :type structure: struc.Structure
    :param atom: Index of the atom in the structure.
    :type atom: int
    :param cutoffs: 2- and 3-body cutoff radii. 2-body if one cutoff is
     given, 2+3-body if two are passed.
    :type cutoffs: np.ndarray
    """

    def __init__(self, structure: Structure, atom: int, cutoffs):
        self.structure = structure
        self.positions = structure.wrapped_positions
        self.cell = structure.cell
        self.species = structure.coded_species

        self.atom = atom
        self.ctype = structure.coded_species[atom]

        self.cutoffs = np.copy(cutoffs)

        # get 2-body arrays
        bond_array_2, bond_positions_2, etypes = \
            get_2_body_arrays(self.positions, self.atom, self.cell,
                              self.cutoffs[0], self.species)
        self.bond_array_2 = bond_array_2
        self.etypes = etypes

        # if 2 cutoffs are given, create 3-body arrays
        if len(cutoffs) > 1:
            bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts = \
                get_3_body_arrays(bond_array_2, bond_positions_2, self.cutoffs[1])
            self.bond_array_3 = bond_array_3
            self.cross_bond_inds = cross_bond_inds
            self.cross_bond_dists = cross_bond_dists
            self.triplet_counts = triplet_counts

        # if 3 cutoffs are given, create many-body arrays
        if len(cutoffs) > 2:
            self.bond_array_mb, self.neigh_dists_mb, self.num_neighs_mb, self.etype_mb = get_m_body_arrays(
                self.positions, self.atom, self.cell, self.cutoffs[2], self.species)
        else:
            self.bond_array_mb = None
            self.neigh_dists_mb = None
            self.num_neighs_mb = None
            self.etype_mb = None


    def as_dict(self):
        """
        Returns Atomic Environment object as a dictionary for serialization
        purposes. Does not include the structure to avoid redundant
        information.
        :return:
        """
        # TODO write serialization method for structure
        # so that the removal of the structure is not messed up
        # by JSON serialization
        dictionary = dict(vars(self))
        dictionary['object'] = 'AtomicEnvironment'
        dictionary['forces'] = self.structure.forces
        dictionary['energy'] = self.structure.energy
        dictionary['stress'] = self.structure.stress

        del dictionary['structure']

        return dictionary

    @staticmethod
    def from_dict(dictionary):
        """
        Loads in atomic environment object from a dictionary which was
        serialized by the to_dict method.

        :param dictionary: Dictionary describing atomic environment.
        """
        # TODO Instead of re-computing 2 and 3 body environment,
        # directly load in, this would be much more efficient

        struc = Structure(cell=np.array(dictionary['cell']),
                          positions=dictionary['positions'],
                          species=dictionary['species'])
        index = dictionary['atom']

        cutoffs = dictionary['cutoffs']

        return AtomicEnvironment(struc, index, cutoffs)

    def __str__(self):
        atom_type = self.ctype
        neighbor_types = self.etypes
        n_neighbors = len(self.bond_array_2)
        string = 'Atomic Env. of Type {} surrounded by {} atoms of Types {}' \
                 ''.format(atom_type, n_neighbors,
                           sorted(list(set(neighbor_types))))

        return string


@njit
def get_2_body_arrays(positions, atom: int, cell, cutoff_2: float, species):
    """Returns distances, coordinates, and species of atoms in the 2-body
    local environment. This method is implemented outside the AtomicEnvironment
    class to allow for njit acceleration with Numba.

    :param positions: Positions of atoms in the structure.
    :type positions: np.ndarray
    :param atom: Index of the central atom of the local environment.
    :type atom: int
    :param cell: 3x3 array whose rows are the Bravais lattice vectors of the
        cell.
    :type cell: np.ndarray
    :param cutoff_2: 2-body cutoff radius.
    :type cutoff_2: float
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :return: Tuple of arrays describing pairs of atoms in the 2-body local
     environment.

     bond_array_2: Array containing the distances and relative
     coordinates of atoms in the 2-body local environment. First column
     contains distances, remaining columns contain Cartesian coordinates
     divided by the distance (with the origin defined as the position of the
     central atom). The rows are sorted by distance from the central atom.

     bond_positions_2: Coordinates of atoms in the 2-body local environment.

     etypes: Species of atoms in the 2-body local environment represented by
     their atomic number.
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """
    noa = len(positions)
    pos_atom = positions[atom]
    coords = np.zeros((noa, 3, 27))
    dists = np.zeros((noa, 27))
    cutoff_count = 0
    super_sweep = np.array([-1, 0, 1])

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        for s1 in super_sweep:
            for s2 in super_sweep:
                for s3 in super_sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
                    if (dist < cutoff_2) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_array_2 = np.zeros((cutoff_count, 4))
    bond_positions_2 = np.zeros((cutoff_count, 3))
    etypes = np.zeros(cutoff_count, dtype=np.int8)
    bond_count = 0

    for m in range(noa):
        spec_curr = species[m]
        for n in range(27):
            dist_curr = dists[m, n]
            if (dist_curr < cutoff_2) and (dist_curr != 0):
                coord = coords[m, :, n]
                bond_array_2[bond_count, 0] = dist_curr
                bond_array_2[bond_count, 1:4] = coord / dist_curr
                bond_positions_2[bond_count, :] = coord
                etypes[bond_count] = spec_curr
                bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]
    etypes = etypes[sort_inds]

    return bond_array_2, bond_positions_2, etypes


@njit
def get_2_body_arrays_ind(positions, atom: int, cell, cutoff_2: float, species):
    """Returns distances, coordinates, species of atoms, and indexes of neighbors
    in the 2-body local environment. This method is implemented outside
    the AtomicEnvironment class to allow for njit acceleration with Numba.

    :param positions: Positions of atoms in the structure.
    :type positions: np.ndarray
    :param atom: Index of the central atom of the local environment.
    :type atom: int
    :param cell: 3x3 array whose rows are the Bravais lattice vectors of the
        cell.
    :type cell: np.ndarray
    :param cutoff_2: 2-body cutoff radius.
    :type cutoff_2: float
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :return: Tuple of arrays describing pairs of atoms in the 2-body local
     environment.

     bond_array_2: Array containing the distances and relative
     coordinates of atoms in the 2-body local environment. First column
     contains distances, remaining columns contain Cartesian coordinates
     divided by the distance (with the origin defined as the position of the
     central atom). The rows are sorted by distance from the central atom.

     bond_positions_2: Coordinates of atoms in the 2-body local environment.

     etypes: Species of atoms in the 2-body local environment represented by
     their atomic number.

     bond_indexes: Structure indexes of atoms in the local environment.

    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    noa = len(positions)
    pos_atom = positions[atom]
    coords = np.zeros((noa, 3, 27))
    dists = np.zeros((noa, 27))
    cutoff_count = 0
    super_sweep = np.array([-1, 0, 1])

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        for s1 in super_sweep:
            for s2 in super_sweep:
                for s3 in super_sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
                    if (dist < cutoff_2) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_indexes = np.zeros(cutoff_count, dtype=np.int8)
    bond_array_2 = np.zeros((cutoff_count, 4))
    bond_positions_2 = np.zeros((cutoff_count, 3))
    etypes = np.zeros(cutoff_count, dtype=np.int8)
    bond_count = 0

    for m in range(noa):
        spec_curr = species[m]
        for n in range(27):
            dist_curr = dists[m, n]
            if (dist_curr < cutoff_2) and (dist_curr != 0):
                coord = coords[m, :, n]
                bond_array_2[bond_count, 0] = dist_curr
                bond_array_2[bond_count, 1:4] = coord / dist_curr
                bond_positions_2[bond_count, :] = coord
                etypes[bond_count] = spec_curr
                bond_indexes[bond_count] = m
                bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]
    bond_indexes = bond_indexes[sort_inds]
    etypes = etypes[sort_inds]

    return bond_array_2, bond_positions_2, etypes, bond_indexes


@njit
def get_3_body_arrays(bond_array_2, bond_positions_2, cutoff_3: float):
    """Returns distances and coordinates of triplets of atoms in the
    3-body local environment.

    :param bond_array_2: 2-body bond array.
    :type bond_array_2: np.ndarray
    :param bond_positions_2: Coordinates of atoms in the 2-body local
     environment.
    :type bond_positions_2: np.ndarray
    :param cutoff_3: 3-body cutoff radius.
    :type cutoff_3: float
    :return: Tuple of 4 arrays describing triplets of atoms in the 3-body local
     environment.

     bond_array_3: Array containing the distances and relative
     coordinates of atoms in the 3-body local environment. First column
     contains distances, remaining columns contain Cartesian coordinates
     divided by the distance (with the origin defined as the position of the
     central atom). The rows are sorted by distance from the central atom.

     cross_bond_inds: Two dimensional array whose row m contains the indices
     of atoms n > m that are within a distance cutoff_3 of both atom n and the
     central atom.

     cross_bond_dists: Two dimensional array whose row m contains the
     distances from atom m of atoms n > m that are within a distance cutoff_3
     of both atom n and the central atom.

     triplet_counts: One dimensional array of integers whose entry m is the
     number of atoms that are within a distance cutoff_3 of atom m.

    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    # get 3-body bond array
    ind_3 = -1
    noa = bond_array_2.shape[0]
    for count, dist in enumerate(bond_array_2[:, 0]):
        if dist > cutoff_3:
            ind_3 = count
            break
    if ind_3 == -1:
        ind_3 = noa

    bond_array_3 = bond_array_2[0:ind_3, :]
    bond_positions_3 = bond_positions_2[0:ind_3, :]

    # get cross bond array
    cross_bond_inds = np.zeros((ind_3, ind_3), dtype=np.int8) - 1
    cross_bond_dists = np.zeros((ind_3, ind_3))
    triplet_counts = np.zeros(ind_3, dtype=np.int8)
    for m in range(ind_3):
        pos1 = bond_positions_3[m]
        count = m + 1
        trips = 0
        for n in range(m + 1, ind_3):
            pos2 = bond_positions_3[n]
            diff = pos2 - pos1
            dist_curr = sqrt(
                diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

            if dist_curr < cutoff_3:
                cross_bond_inds[m, count] = n
                cross_bond_dists[m, count] = dist_curr
                count += 1
                trips += 1
        triplet_counts[m] = trips

    return bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts


@njit
def get_m_body_arrays(positions, atom: int, cell, cutoff_mb: float, species):
    """Returns distances, and species of atoms in the many-body
    local environment, and returns distances and numbers of neighbours for atoms in the one
    many-body local environment. This method is implemented outside the AtomicEnvironment
    class to allow for njit acceleration with Numba.

    :param positions: Positions of atoms in the structure.
    :type positions: np.ndarray
    :param atom: Index of the central atom of the local environment.
    :type atom: int
    :param cell: 3x3 array whose rows are the Bravais lattice vectors of the
        cell.
    :type cell: np.ndarray
    :param cutoff_mb: 2-body cutoff radius.
    :type cutoff_mb: float
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :param indexes: Boolean indicating whether indexes of neighbours are returned
    :type indexes: boolean
    :return: Tuple of arrays describing pairs of atoms in the 2-body local
     environment.

     bond_array_mb: Array containing the distances and relative
     coordinates of atoms in the 2-body local environment. First column
     contains distances, remaining columns contain Cartesian coordinates
     divided by the distance (with the origin defined as the position of the
     central atom). The rows are sorted by distance from the central atom.

     etypes: Species of atoms in the 2-body local environment represented by
     their atomic number.

     neigh_dists_mb: Matrix padded with zero values of distances
     of neighbours for the atoms in the local environment.

     num_neighs_mb: number of neighbours of each atom in the local environment

     etypes_mb_array: species of neighbours of each atom in the local environment

    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    # TODO: this can be probably improved using stored arrays, redundant calls to get_2_body_arrays
    # Get distances, positions, species and indexes of neighbouring atoms
    bond_array_mb, __, _, bond_inds = get_2_body_arrays_ind(
        positions, atom, cell, cutoff_mb, species)

    # For each neighbouring atom, get distances in its neighbourhood
    neighbouring_dists = []
    neighbouring_etypes = []
    max_neighbours = 0
    for m in bond_inds:
        neighbour_bond_array_2, ___, etypes_mb = get_2_body_arrays(positions, m, cell,
                                                         cutoff_mb, species)
        neighbouring_dists.append(neighbour_bond_array_2[:, 0])
        neighbouring_etypes.append(etypes_mb)
        if len(neighbour_bond_array_2[:, 0]) > max_neighbours:
            max_neighbours = len(neighbour_bond_array_2[:, 0])

    # Transform list of distances into Numpy array
    neigh_dists_mb = np.zeros((len(bond_inds), max_neighbours))
    num_neighs_mb = np.zeros(len(bond_inds), dtype=np.int8)
    etypes_mb_array = np.zeros((len(bond_inds), max_neighbours), dtype=np.int8)
    for i in range(len(bond_inds)):
        num_neighs_mb[i] = len(neighbouring_dists[i])
        neigh_dists_mb[i, :num_neighs_mb[i]] = neighbouring_dists[i]
        etypes_mb_array[i, :num_neighs_mb[i]] = neighbouring_etypes[i]


    return bond_array_mb, neigh_dists_mb, num_neighs_mb, etypes_mb_array


if __name__ == '__main__':
    pass
