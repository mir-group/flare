from math import sqrt
from numba import njit
import numpy as np
import flare.kernels.cutoffs as cf
from flare.kernels.kernels import coordination_number, q_value_mc

@njit
def get_2_body_arrays(positions, atom: int, cell, r_cut, cutoff_2, species, sweep,
                      nspecie, specie_mask, twobody_mask):
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
    :param twobody_mask: mapping from the types of end atoms to bond types
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
    super_count = sweep.shape[0]**3
    coords = np.zeros((noa, 3, super_count), dtype=np.float64)
    dists = np.zeros((noa, super_count), dtype=np.float64)
    cutoff_count = 0

    vec1 = cell[0]
    vec2 = cell[1]
    vec3 = cell[2]

    sepcut = False
    bcn = 0
    if nspecie > 1 and cutoff_2 is not None:
        sepcut = True
        bc = specie_mask[species[atom]]
        bcn = nspecie * bc

    # record distances and positions of images
    for n in range(noa):
        diff_curr = positions[n] - pos_atom
        im_count = 0
        if sepcut and (specie_mask is not None) and (cutoff_2 is not None):
            bn = specie_mask[species[n]]
            r_cut = cutoff_2[twobody_mask[bn+bcn]]

        for s1 in sweep:
            for s2 in sweep:
                for s3 in sweep:
                    im = diff_curr + s1 * vec1 + s2 * vec2 + s3 * vec3
                    dist = sqrt(im[0] * im[0] + im[1] * im[1] + im[2] * im[2])
                    if (dist < r_cut) and (dist != 0):
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
        if sepcut and (specie_mask is not None) and (cutoff_2 is not None):
            bm = specie_mask[species[m]]
            r_cut = cutoff_2[twobody_mask[bm+bcn]]
        for im_count in range(super_count):
            dist_curr = dists[m, im_count]
            if (dist_curr < r_cut) and (dist_curr != 0):
                coord = coords[m, :, im_count]
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
def get_3_body_arrays(bond_array_2, bond_positions_2, ctype,
                      etypes, r_cut, cutoff_3,
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

    sepcut = False
    if nspecie > 1 and cutoff_3 is not None:
        bc = specie_mask[ctype]
        bcn = nspecie * bc
        r_cut = np.max(cutoff_3)
        sepcut = True

    # get 3-body bond array
    ind_3_l = np.where(bond_array_2[:, 0] > r_cut)[0]
    if (ind_3_l.shape[0] > 0):
        ind_3 = ind_3_l[0]
    else:
        ind_3 = bond_array_2.shape[0]

    bond_array_3 = bond_array_2[0:ind_3, :]
    bond_positions_3 = bond_positions_2[0:ind_3, :]

    cut_m = r_cut
    cut_n = r_cut
    cut_mn = r_cut

    # get cross bond array
    cross_bond_inds = np.zeros((ind_3, ind_3), dtype=np.int8) - 1
    cross_bond_dists = np.zeros((ind_3, ind_3), dtype=np.float64)
    triplet_counts = np.zeros(ind_3, dtype=np.int8)
    for m in range(ind_3):
        pos1 = bond_positions_3[m]
        count = m + 1
        trips = 0

        if sepcut and (specie_mask is not None) and (cut3b_mask is not None) and (cutoff_3 is not None):
            # choose bond dependent bond
            bm = specie_mask[etypes[m]]
            btype_m = cut3b_mask[bm + bcn]  # (m, c)
            cut_m = cutoff_3[btype_m]
            bmn = nspecie * bm  # for cross_dist usage

        for n in range(m + 1, ind_3):

            if sepcut and (specie_mask is not None) and (cut3b_mask is not None) and (cutoff_3 is not None):
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
def get_m2_body_arrays(positions, atom: int, cell, r_cut, manybody_cutoff_list,
    species, sweep: np.ndarray, nspec, spec_mask, manybody_mask,
    cutoff_func=cf.quadratic_cutoff):
    # TODO:
    # 1. need to deal with the conflict of cutoff functions if other funcs are used
    # 2. complete the docs of "Return"
    # TODO: this can be probably improved using stored arrays, redundant calls to get_2_body_arrays
    # Get distances, positions, species and indices of neighbouring atoms
    """
    Args:
        positions (np.ndarray): Positions of atoms in the structure.
        atom (int): Index of the central atom of the local environment.
        cell (np.ndarray): 3x3 array whose rows are the Bravais lattice vectors of the
            cell.
        manybody_cutoff_list (float): 2-body cutoff radius.
        species (np.ndarray): Numpy array of species represented by their atomic numbers.

    Return:
        Tuple of arrays describing pairs of atoms in the 2-body local
        environment.
    """
    # Get distances, positions, species and indexes of neighbouring atoms
    bond_array_mb, _, etypes, bond_inds = get_2_body_arrays(
        positions, atom, cell, r_cut, manybody_cutoff_list, species, sweep,
        nspec, spec_mask, manybody_mask)

    sepcut = False
    if nspec > 1 and manybody_cutoff_list is not None:
        bc = spec_mask[species[atom]]
        bcn = bc * nspec
        sepcut = True

    species_list = np.array(list(set(species)), dtype=np.int8)
    n_bonds = len(bond_inds)
    n_specs = len(species_list)
    qs = np.zeros(n_specs, dtype=np.float64)
    qs_neigh = np.zeros((n_bonds, n_specs), dtype=np.float64)
    q_neigh_grads = np.zeros((n_bonds, 3), dtype=np.float64)

    # get coordination number of center atom for each species
    for s in range(n_specs):
        if sepcut and (spec_mask is not None) and (manybody_mask is not None) and (manybody_cutoff_list is not None):
            bs = spec_mask[species_list[s]]
            mbtype = manybody_mask[bcn + bs]
            r_cut = manybody_cutoff_list[mbtype]

        qs[s] = q_value_mc(bond_array_mb[:, 0], r_cut, species_list[s],
                           etypes, cutoff_func)

    # get coordination number of all neighbor atoms for each species
    for i in range(n_bonds):
        if sepcut and (spec_mask is not None) and (manybody_mask is not None) and (manybody_cutoff_list is not None):
            be = spec_mask[etypes[i]]
            ben = be * nspec

        neigh_bond_array, __, neigh_etypes, ___ = \
            get_2_body_arrays(positions, bond_inds[i], cell, r_cut,
                              manybody_cutoff_list, species, sweep, nspec, spec_mask, manybody_mask)
        for s in range(n_specs):
            if sepcut and (spec_mask is not None) and (manybody_mask is not None) and (manybody_cutoff_list is not None):
                bs = spec_mask[species_list[s]]
                mbtype = manybody_mask[bs + ben]
                r_cut = manybody_cutoff_list[mbtype]

            qs_neigh[i, s] = q_value_mc(neigh_bond_array[:, 0], r_cut,
                                        species_list[s], neigh_etypes, cutoff_func)

    # get grad from each neighbor atom
    for i in range(n_bonds):
        if sepcut and (spec_mask is not None) and (manybody_mask is not None) and (manybody_cutoff_list is not None):
            be = spec_mask[etypes[i]]
            mbtype = manybody_mask[bcn + be]
            r_cut = manybody_cutoff_list[mbtype]

        ri = bond_array_mb[i, 0]
        for d in range(3):
            ci = bond_array_mb[i, d+1]

            ____, q_neigh_grads[i, d] = coordination_number(ri, ci, r_cut, 
                cutoff_func)

    # get grads of the center atom
    q_grads =  q2_grads_mc(q_neigh_grads, species_list, etypes)

    return qs, qs_neigh, q_grads, q_neigh_grads, species_list, etypes 

@njit
def q2_grads_mc(neigh_grads, species_list, etypes):
    n_specs = len(species_list)
    n_neigh = neigh_grads.shape[0]
    grads = np.zeros((n_specs, 3))
    for i in range(n_neigh):
        si = np.where(species_list==etypes[i])[0][0]
        grads[si, :] += neigh_grads[i, :]

    return grads


@njit
def get_m3_body_arrays(positions, atom: int, cell, cutoff: float, species, 
                       sweep, cutoff_func=cf.quadratic_cutoff):
    """
    Note: here we assume the cutoff is not too large, 
    i.e., 2 * cutoff < cell_size 
    """
    species_list = np.array(list(set(species)), dtype=np.int8)

    q_func = coordination_number

    bond_array, bond_positions, etypes, bond_inds = \
        get_2_body_arrays(positions, atom, cell, cutoff, species, sweep)

    bond_array_m3b, cross_bond_inds, cross_bond_dists, triplets = \
        get_3_body_arrays(bond_array, bond_positions, cutoff)

    # get descriptor of center atom for each species
    m3b_array = q3_value_mc(bond_array_m3b[:, 0], cross_bond_inds, 
        cross_bond_dists, triplets, cutoff, species_list, etypes, 
        cutoff_func, q_func)


    # get descriptor of all neighbor atoms for each species
    n_bonds = len(bond_array_m3b)
    n_specs = len(species_list)
    m3b_neigh_array = np.zeros((n_bonds, n_specs, n_specs))
    for i in range(n_bonds):
        neigh_bond_array, neigh_positions, neigh_etypes, _ = \
            get_2_body_arrays(positions, bond_inds[i], cell, cutoff, species, sweep)

        neigh_array_m3b, neigh_cross_inds, neigh_cross_dists, neigh_triplets = \
            get_3_body_arrays(neigh_bond_array, neigh_positions, cutoff)

        m3b_neigh_array[i, :, :] = q3_value_mc(neigh_array_m3b[:, 0],
            neigh_cross_inds, neigh_cross_dists, neigh_triplets, 
            cutoff, species_list, neigh_etypes, cutoff_func, q_func)

    # get grad from each neighbor atom, assume the cutoff is not too large
    # such that 2 * cutoff < cell_size 
    m3b_neigh_grads = q3_neigh_grads_mc(bond_array_m3b, cross_bond_inds, 
        cross_bond_dists, triplets, cutoff, species_list, etypes, 
        cutoff_func, q_func)

    # get grads of the center atom
    m3b_grads = q3_grads_mc(m3b_neigh_grads, species_list, etypes)

    return m3b_array, m3b_neigh_array, m3b_grads, m3b_neigh_grads, species_list, etypes

@njit
def q3_grads_mc(neigh_grads, species_list, etypes):
    n_specs = len(species_list)
    n_neigh = neigh_grads.shape[0]
    grads = np.zeros((n_specs, n_specs, 3))
    for i in range(n_neigh):
        si = np.where(species_list==etypes[i])[0][0]
        for spec_j in species_list:
            sj = np.where(species_list==spec_j)[0][0]
            if si == sj:
                grads[si, sj, :] += neigh_grads[i, sj, :] / 2
            else:
                grads[si, sj, :] += neigh_grads[i, sj, :]

    return grads

@njit
def q3_neigh_grads_mc(bond_array_m3b, cross_bond_inds, cross_bond_dists, 
    triplets, r_cut, species_list, etypes, cutoff_func, 
    q_func=coordination_number):

    n_bonds = len(bond_array_m3b)
    n_specs = len(species_list)
    m3b_grads = np.zeros((n_bonds, n_specs, 3))

    # get grad from each neighbor atom
    for i in range(n_bonds):

        # get grad of q_func
        ri = bond_array_m3b[i, 0]
        si = np.where(species_list==etypes[i])[0][0]
        qi, _ = q_func(ri, 0, r_cut, cutoff_func)

        qi_grads = np.zeros(3)
        for d in range(3):
            ci = bond_array_m3b[i, d + 1]
            _, qi_grads[d] = q_func(ri, ci, r_cut, cutoff_func)

        # go through all triplets with "atom" and "i"
        for ind in range(triplets[i]): 
            j = cross_bond_inds[i, i + ind + 1]
            rj = bond_array_m3b[j, 0]
            sj = np.where(species_list==etypes[j])[0][0]
            qj, _ = q_func(rj, 0, r_cut, cutoff_func)
            
            qj_grads = np.zeros(3)
            for d in range(3):
                cj = bond_array_m3b[j, d + 1]
                _, qj_grads[d] = q_func(rj, cj, r_cut, cutoff_func)

            rij = cross_bond_dists[i, i + ind + 1] 
            qij, _ = q_func(rij, 0, r_cut, cutoff_func)

            q_grad = (qi_grads * qj + qi * qj_grads) * qij

            # remove duplicant
    #        if si == sj:
    #            q_grad /= 2
            m3b_grads[i, sj, :] += q_grad
            m3b_grads[j, si, :] += q_grad

    return m3b_grads


@njit
def q3_value_mc(distances, cross_bond_inds, cross_bond_dists, triplets,
    r_cut, species_list, etypes, cutoff_func, q_func=coordination_number):
    """Compute value of many-body many components descriptor based
    on distances of atoms in the local many-body environment.

    Args:
        distances (np.ndarray): distances between atoms i and j
        r_cut (float): cutoff hyperparameter
        ref_species (int): species to consider to compute the contribution
        etypes (np.ndarray): atomic species of neighbours
        cutoff_func (callable): cutoff function
        q_func (callable): many-body pairwise descrptor function

    Return:
        float: the value of the many-body descriptor
    """
    n_specs = len(species_list)
    mb3_array = np.zeros((n_specs, n_specs))
    n_bonds = len(distances)

    for m in range(n_bonds):
        q1, _ = q_func(distances[m], 0, r_cut, cutoff_func)
        s1 = np.where(species_list==etypes[m])[0][0]

        for n in range(triplets[m]):
            ind = cross_bond_inds[m, m + n + 1]
            s2 = np.where(species_list==etypes[ind])[0][0] 
            q2, _ = q_func(distances[ind], 0, r_cut, cutoff_func)
    
            r3 = cross_bond_dists[m, m + n + 1]
            q3, _ = q_func(r3, 0, r_cut, cutoff_func)

            mb3_array[s1, s2] += q1 * q2 * q3
            if s1 != s2:
                mb3_array[s2, s1] += q1 * q2 * q3

    return mb3_array 

