"""The :class:`AtomicEnvironment` object stores information about the local
environment of an atom. :class:`AtomicEnvironment` objects are inputs to the
2-, 3-, and 2+3-body kernels."""
import numpy as np
from math import sqrt
from numba import njit
from flare.struc import Structure
from flare.mask_helper import HyperParameterMasking


class AtomicEnvironment:
    """Contains information about the local environment of an atom,
    including arrays of pair and triplet distances and the chemical
    species of atoms in the environment.

    :param structure: Structure of atoms.
    :type structure: struc.Structure
    :param atom: Index of the atom in the structure.
    :type atom: int
    :param cutoffs: 2- and 3-body cutoff radii. 2-body if one cutoff is
    given, 2+3-body if two are passed.
    :type cutoffs: np.ndarray
    :param cutoffs_mask: a dictionary to store multiple cutoffs if neede
                         it should be exactly the same as the hyps mask
    :type cutoffs_mask: dict

    The cutoffs_mask allows the user to define multiple cutoffs for different
    bonds, triples, and many body interaction. This dictionary should be
    consistent with the hyps_mask used in the GuassianProcess object.

    * specie_mask: 118-long integer array descirbing which elements belong to
                 like groups for determining which bond hyperparameters to use.
                 For instance, [0,0,1,1,0 ...] assigns H to group 0, He and
                 Li to group 1, and Be to group 0 (the 0th register is ignored).
    * nspecie: Integer, number of different species groups (equal to number of
             unique values in specie_mask).
    * nbond: Integer, number of different hyperparameter/cutoff sets to associate with
             different 2-body pairings of atoms in groups defined in specie_mask.
    * bond_mask: Array of length nspecie^2, which describes the cutoff to
                 associate with different pairings of species types. For example, if there
                 are atoms of type 0 and 1, then bond_mask defines which cutoff
                 to use for parings [0-0, 0-1, 1-0, 1-1]: if we wanted cutoff0 for
                 0-0 parings and set 1 for 0-1 and 1-1 pairings, then we would make
                 bond_mask [0, 1, 1, 1].
    * cutoff_2b: Array of length nbond, which stores the cutoff used for different
                 types of bonds defined in bond_mask
    * ncut3b:    Integer, number of different cutoffs sets to associate
                 with different 3-body pariings of atoms in groups defined in specie_mask.
    * cut3b_mask: Array of length nspecie^2, which describes the cutoff to
                 associate with different bond types in triplets. For example, in a triplet
                 (C, O, H) , there are three cutoffs. Cutoffs for CH bond, CO bond and OH bond.
                 If C and O are associate with atom group 1 in specie_mask and H are associate with
                 group 0 in specie_mask, the cut3b_mask[1*nspecie+0] determines the C/O-H bond cutoff,
                 and cut3b_mask[1*nspecie+1] determines the C-O bond cutoff. If we want the
                 former one to use the 1st cutoff in cutoff_3b and the later to use the 2nd cutoff
                 in cutoff_3b, the cut3b_mask should be [0, 0, 0, 1]
    * cutoff_3b: Array of length ncut3b, which stores the cutoff used for different
                 types of bonds in triplets.
    * nmb :      Integer, number of different cutoffs set to associate with different coordination
                 numbers
    * mb_mask:   similar to bond_mask and cut3b_mask.
    * cutoff_mb: Array of length nmb, stores the cutoff used for different many body terms

    Examples can be found at the end of in tests/test_env.py

    """

    def __init__(self, structure: Structure, atom: int, cutoffs, sweep=1, cutoffs_mask=None):
        self.structure = structure
        self.positions = structure.wrapped_positions
        self.cell = structure.cell
        self.species = structure.coded_species
        self.sweep_array = np.arange(-sweep, sweep+1, 1)

        self.atom = atom
        self.ctype = structure.coded_species[atom]

        self.cutoffs = np.copy(cutoffs)
        self.cutoffs_mask = cutoffs_mask

        self.setup_mask()

        assert self.scalar_cutoff_3 <= self.scalar_cutoff_2, \
            "2b cutoff has to be larger than 3b cutoff"
        # # TO DO, once the mb function is updated to use the bond_array_2
        # # this block should be activated.
        # assert self.scalar_cutoff_mb <= self.scalar_cutoff_2, \
        #         "mb cutoff has to be larger than mb cutoff"

        self.compute_env()

    def setup_mask(self):

        self.scalar_cutoff_2, self.scalar_cutoff_3, self.scalar_cutoff_mb, \
            self.cutoff_2b, self.cutoff_3b, self.cutoff_mb, \
            self.nspecie, self.n2b, self.n3b, self.nmb, self.specie_mask, \
            self.bond_mask, self.cut3b_mask, self.mb_mask = \
            HyperParameterMasking.mask2cutoff(self.cutoffs, self.cutoffs_mask)

    def compute_env(self):

        # get 2-body arrays
        if (self.n2b > 1):
            bond_array_2, bond_positions_2, etypes, bond_inds = \
                get_2_body_arrays_sepcut(self.positions, self.atom, self.cell,
                                         self.cutoff_2b, self.species, self.sweep_array,
                                         self.nspecie, self.specie_mask, self.bond_mask)
        else:
            bond_array_2, bond_positions_2, etypes, bond_inds = \
                get_2_body_arrays(self.positions, self.atom, self.cell,
                                  self.scalar_cutoff_2, self.species, self.sweep_array)

        self.bond_array_2 = bond_array_2
        self.etypes = etypes
        self.bond_inds = bond_inds

        # if 2 cutoffs are given, create 3-body arrays
        if self.scalar_cutoff_3 > 0:
            if (self.n3b > 1):
                bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts = \
                    get_3_body_arrays_sepcut(bond_array_2, bond_positions_2,
                                             self.species[self.atom], etypes, self.cutoff_3b,
                                             self.nspecie, self.specie_mask, self.cut3b_mask)
            else:
                bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts = \
                    get_3_body_arrays(
                        bond_array_2, bond_positions_2, self.scalar_cutoff_3)

            self.bond_array_3 = bond_array_3
            self.cross_bond_inds = cross_bond_inds
            self.cross_bond_dists = cross_bond_dists
            self.triplet_counts = triplet_counts

        # if 3 cutoffs are given, create many-body arrays
        if self.scalar_cutoff_mb > 0:
            if (self.nmb > 1):
                self.bond_array_mb, self.neigh_dists_mb, \
                    self.num_neighs_mb, self.etype_mb, \
                    self.bond_array_mb_etypes = \
                    get_m_body_arrays_sepcut(
                        self.positions, self.atom, self.cell,
                        self.cutoff_mb, self.species, self.sweep_array,
                        self.nspecie, self.specie_mask, self.mb_mask)
            else:
                self.bond_array_mb, self.neigh_dists_mb, \
                    self.num_neighs_mb, self.etype_mb, \
                    self.bond_array_mb_etypes = \
                    get_m_body_arrays(self.positions, self.atom,
                                      self.cell, self.scalar_cutoff_mb,
                                      self.species, self.sweep_array)
        else:
            self.bond_array_mb = None
            self.neigh_dists_mb = None
            self.num_neighs_mb = None
            self.etype_mb = None
            self.bond_array_mb_etypes = None

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
        dictionary['cutoffs_mask'] = self.cutoffs_mask

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

        if dictionary.get('cutoffs') is not None:
            cutoffs = dictionary['cutoffs']
        else:
            cutoffs = {}
            for cutoff_type in ['2','3','mb']:
                key = 'scalar_cutoff_'+cutoff_type
                if (key in dictionary):
                    cutoffs[key] = dictionary[key]

        cutoffs_mask = dictionary.get('cutoffs_mask', None)

        return AtomicEnvironment(struc, index, cutoffs, cutoffs_mask=cutoffs_mask)

    def __str__(self):
        atom_type = self.ctype
        neighbor_types = self.etypes
        n_neighbors = len(self.bond_array_2)
        string = 'Atomic Env. of Type {} surrounded by {} atoms '\
                 'of Types {}'.format(atom_type, n_neighbors,
                                      sorted(list(set(neighbor_types))))

        return string


@njit
def get_2_body_arrays(positions: np.ndarray, atom: int, cell: np.ndarray,
                      cutoff_2: float, species: np.ndarray, sweep: np.ndarray):
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

     bond_indices: Structure indices of atoms in the local environment.

    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """

    noa = len(positions)
    pos_atom = positions[atom]
    super_count = sweep.shape[0]**3
    coords = np.zeros((noa, 3, super_count))
    dists = np.zeros((noa, super_count))
    cutoff_count = 0

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        for s1 in sweep:
            for s2 in sweep:
                for s3 in sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1]
                                + im[2] * im[2])
                    if (dist < cutoff_2) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_indices = np.zeros(cutoff_count, dtype=np.int8)
    bond_array_2 = np.zeros((cutoff_count, 4), dtype=np.float64)
    bond_positions_2 = np.zeros((cutoff_count, 3), dtype=np.float64)
    etypes = np.zeros(cutoff_count, dtype=np.int8)
    bond_count = 0

    for m in range(noa):
        spec_curr = species[m]
        for n in range(super_count):
            dist_curr = dists[m, n]
            if (dist_curr < cutoff_2) and (dist_curr != 0):
                coord = coords[m, :, n]
                bond_array_2[bond_count, 0] = dist_curr
                bond_array_2[bond_count, 1:4] = coord / dist_curr
                bond_positions_2[bond_count, :] = coord
                etypes[bond_count] = spec_curr
                bond_indices[bond_count] = m
                bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]
    bond_indices = bond_indices[sort_inds]
    etypes = etypes[sort_inds]

    return bond_array_2, bond_positions_2, etypes, bond_indices


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
def get_2_body_arrays_sepcut(positions, atom: int, cell, cutoff_2, species, sweep,
                             nspecie, specie_mask, bond_mask):
    """Returns distances, coordinates, species of atoms, and indices of neighbors
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
    :type cutoff_2: np.ndarray
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :param nspecie: number of atom types to define bonds
    :type: int
    :param specie_mask: mapping from atomic number to atom types
    :type: np.ndarray
    :param bond_mask: mapping from the types of end atoms to bond types
    :type: np.ndarray
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

     bond_indices: Structure indices of atoms in the local environment.

    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    noa = len(positions)
    pos_atom = positions[atom]
    coords = np.zeros((noa, 3, 27), dtype=np.float64)
    dists = np.zeros((noa, 27), dtype=np.float64)
    cutoff_count = 0

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    bc = specie_mask[species[atom]]
    bcn = nspecie * bc

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        bn = specie_mask[species[n]]
        rcut = cutoff_2[bond_mask[bn+bcn]]

        for s1 in sweep:
            for s2 in sweep:
                for s3 in sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
                    if (dist < rcut) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_indices = np.zeros(cutoff_count, dtype=np.int8)
    bond_array_2 = np.zeros((cutoff_count, 4), dtype=np.float64)
    bond_positions_2 = np.zeros((cutoff_count, 3), dtype=np.float64)
    etypes = np.zeros(cutoff_count, dtype=np.int8)
    bond_count = 0

    for m in range(noa):
        spec_curr = species[m]
        bm = specie_mask[species[m]]
        rcut = cutoff_2[bond_mask[bm+bcn]]
        for n in range(27):
            dist_curr = dists[m, n]
            if (dist_curr < rcut) and (dist_curr != 0):
                coord = coords[m, :, n]
                bond_array_2[bond_count, 0] = dist_curr
                bond_array_2[bond_count, 1:4] = coord / dist_curr
                bond_positions_2[bond_count, :] = coord
                etypes[bond_count] = spec_curr
                bond_indices[bond_count] = m
                bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]
    bond_indices = bond_indices[sort_inds]
    etypes = etypes[sort_inds]

    return bond_array_2, bond_positions_2, etypes, bond_indices


@njit
def get_3_body_arrays_sepcut(bond_array_2, bond_positions_2, ctype,
                             etypes, cutoff_3,
                             nspecie, specie_mask, cut3b_mask):
    """Returns distances and coordinates of triplets of atoms in the
    3-body local environment.

    :param bond_array_2: 2-body bond array.
    :type bond_array_2: np.ndarray
    :param bond_positions_2: Coordinates of atoms in the 2-body local
     environment.
    :type bond_positions_2: np.ndarray
    :param ctype: atomic number of the center atom
    :type: int
    :param cutoff_3: 3-body cutoff radius.
    :type cutoff_3: np.ndarray
    :param nspecie: number of atom types to define bonds
    :type: int
    :param specie_mask: mapping from atomic number to atom types
    :type: np.ndarray
    :param cut3b_mask: mapping from the types of end atoms to bond types
    :type: np.ndarray
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

    bc = specie_mask[ctype]
    bcn = nspecie * bc

    cut3 = np.max(cutoff_3)

    # get 3-body bond array
    ind_3_l = np.where(bond_array_2[:, 0] > cut3)[0]
    if (ind_3_l.shape[0] > 0):
        ind_3 = ind_3_l[0]
    else:
        ind_3 = bond_array_2.shape[0]

    bond_array_3 = bond_array_2[0:ind_3, :]
    bond_positions_3 = bond_positions_2[0:ind_3, :]

    # get cross bond array
    cross_bond_inds = np.zeros((ind_3, ind_3), dtype=np.int8) - 1
    cross_bond_dists = np.zeros((ind_3, ind_3), dtype=np.float64)
    triplet_counts = np.zeros(ind_3, dtype=np.int8)
    for m in range(ind_3):
        pos1 = bond_positions_3[m]
        count = m + 1
        trips = 0

        # choose bond dependent bond
        bm = specie_mask[etypes[m]]
        btype_m = cut3b_mask[bm + bcn]  # (m, c)
        cut_m = cutoff_3[btype_m]
        bmn = nspecie * bm  # for cross_dist usage

        for n in range(m + 1, ind_3):

            bn = specie_mask[etypes[n]]
            btype_n = cut3b_mask[bn + bcn]  # (n, c)
            cut_n = cutoff_3[btype_n]

            # for cross_dist (m,n) pair
            btype_mn = cut3b_mask[bn + bmn]
            cut_mn = cutoff_3[btype_mn]

            pos2 = bond_positions_3[n]
            diff = pos2 - pos1
            dist_curr = sqrt(
                diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

            if dist_curr < cut_mn \
                    and bond_array_2[m, 0] < cut_m \
                    and bond_array_2[n, 0] < cut_n:
                cross_bond_inds[m, count] = n
                cross_bond_dists[m, count] = dist_curr
                count += 1
                trips += 1

        triplet_counts[m] = trips

    return bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts


@njit
def get_m_body_arrays(positions, atom: int, cell, cutoff_mb: float, species,
                      sweep: np.ndarray):
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
    :type cutoff_mb: np.ndarray
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :param nspecie: number of atom types to define bonds
    :type: int
    :param specie_mask: mapping from atomic number to atom types
    :type: np.ndarray
    :param mb_mask: mapping from the types of end atoms to CN types
    :type: np.ndarray
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
    # Get distances, positions, species and indices of neighbouring atoms
    bond_array_mb, __, etypes, bond_inds = get_2_body_arrays(
        positions, atom, cell, cutoff_mb, species, sweep)

    # For each neighbouring atom, get distances in its neighbourhood
    neighbouring_dists = []
    neighbouring_etypes = []
    max_neighbours = 0
    for m in bond_inds:
        neighbour_bond_array_2, ___, etypes_mb, ____ = \
            get_2_body_arrays(positions, m, cell, cutoff_mb, species, sweep)
        neighbouring_dists.append(neighbour_bond_array_2[:, 0])
        neighbouring_etypes.append(etypes_mb)
        if len(neighbour_bond_array_2[:, 0]) > max_neighbours:
            max_neighbours = len(neighbour_bond_array_2[:, 0])

    # Transform list of distances into Numpy array
    neigh_dists_mb = \
        np.zeros((len(bond_inds), max_neighbours), dtype=np.float64)
    num_neighs_mb = np.zeros(len(bond_inds), dtype=np.int8)
    etypes_mb_array = np.zeros((len(bond_inds), max_neighbours), dtype=np.int8)
    for i in range(len(bond_inds)):
        num_neighs_mb[i] = len(neighbouring_dists[i])
        neigh_dists_mb[i, :num_neighs_mb[i]] = neighbouring_dists[i]
        etypes_mb_array[i, :num_neighs_mb[i]] = neighbouring_etypes[i]

    return bond_array_mb, neigh_dists_mb, num_neighs_mb, etypes_mb_array, etypes

@njit
def get_m_body_arrays_sepcut(positions, atom: int, cell, cutoff_mb, species,
                             sweep: np.ndarray, nspecie, specie_mask, mb_mask):
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
    :type cutoff_mb: np.ndarray
    :param species: Numpy array of species represented by their atomic numbers.
    :type species: np.ndarray
    :param nspecie: number of atom types to define bonds
    :type: int
    :param specie_mask: mapping from atomic number to atom types
    :type: np.ndarray
    :param mb_mask: mapping from the types of end atoms to CN types
    :type: np.ndarray
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
    # Get distances, positions, species and indices of neighbouring atoms
    bond_array_mb, __, etypes, bond_inds = get_2_body_arrays_sepcut(
        positions, atom, cell, cutoff_mb, species, sweep,
        nspecie, specie_mask, mb_mask)

    # For each neighbouring atom, get distances in its neighbourhood
    neighbouring_dists = []
    neighbouring_etypes = []
    max_neighbours = 0
    for m in bond_inds:
        neighbour_bond_array_2, ___, etypes_mb, ____ \
            = get_2_body_arrays_sepcut(positions, m, cell,
                                       cutoff_mb, species, sweep,
                                       nspecie, specie_mask, mb_mask)
        neighbouring_dists.append(neighbour_bond_array_2[:, 0])
        neighbouring_etypes.append(etypes_mb)
        if len(neighbour_bond_array_2[:, 0]) > max_neighbours:
            max_neighbours = len(neighbour_bond_array_2[:, 0])

    # Transform list of distances into Numpy array
    neigh_dists_mb = \
        np.zeros((len(bond_inds), max_neighbours), dtype=np.float64)
    num_neighs_mb = np.zeros(len(bond_inds), dtype=np.int8)
    etypes_mb_array = np.zeros((len(bond_inds), max_neighbours), dtype=np.int8)
    for i in range(len(bond_inds)):
        num_neighs_mb[i] = len(neighbouring_dists[i])
        neigh_dists_mb[i, :num_neighs_mb[i]] = neighbouring_dists[i]
        etypes_mb_array[i, :num_neighs_mb[i]] = neighbouring_etypes[i]

    return bond_array_mb, neigh_dists_mb, num_neighs_mb, etypes_mb_array, etypes


if __name__ == '__main__':
    pass
