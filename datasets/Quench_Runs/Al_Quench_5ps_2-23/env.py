import numpy as np
from math import sqrt
from numba import njit
from struc import Structure


class AtomicEnvironment:
    def __init__(self, structure: Structure, atom: int, cutoffs: np.ndarray):
        self.positions = structure.wrapped_positions
        self.cell = structure.cell
        self.atom = atom
        self.cutoff_2 = cutoffs[0]

        # get 2-body arrays
        bond_array_2, bond_positions_2 = \
            get_2_body_arrays(self.positions, self.atom, self.cell,
                              self.cutoff_2)
        self.bond_array_2 = bond_array_2

        # if multiple cutoffs are given, create 3-body arrays
        if len(cutoffs) > 1:
            self.cutoff_3 = cutoffs[1]
            bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts =\
                get_3_body_arrays(bond_array_2, bond_positions_2, cutoffs[1])
            self.bond_array_3 = bond_array_3
            self.cross_bond_inds = cross_bond_inds
            self.cross_bond_dists = cross_bond_dists
            self.triplet_counts = triplet_counts


@njit
def get_2_body_arrays(positions: np.ndarray, atom: int, cell: np.ndarray,
                      cutoff_2: float):
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
                    im = diff_curr + s1*vec1 + s2*vec2 + s3*vec3
                    dist = sqrt(im[0]*im[0]+im[1]*im[1]+im[2]*im[2])
                    if (dist < cutoff_2) and (dist != 0):
                        dists[n, im_count] = dist
                        coords[n, :, im_count] = im
                        cutoff_count += 1
                    im_count += 1

    # create 2-body bond array
    bond_array_2 = np.zeros((cutoff_count, 4))
    bond_positions_2 = np.zeros((cutoff_count, 3))
    bond_count = 0

    for m in range(noa):
            for n in range(27):
                dist_curr = dists[m, n]
                if (dist_curr < cutoff_2) and (dist_curr != 0):
                    coord = coords[m, :, n]
                    bond_array_2[bond_count, 0] = dist_curr
                    bond_array_2[bond_count, 1:4] = coord / dist_curr
                    bond_positions_2[bond_count, :] = coord
                    bond_count += 1

    # sort by distance
    sort_inds = bond_array_2[:, 0].argsort()
    bond_array_2 = bond_array_2[sort_inds]
    bond_positions_2 = bond_positions_2[sort_inds]

    return bond_array_2, bond_positions_2


@njit
def get_3_body_arrays(bond_array_2: np.ndarray,
                      bond_positions_2: np.ndarray,
                      cutoff_3: float):
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
    cross_bond_inds = np.zeros((ind_3, ind_3), dtype=np.int8)-1
    cross_bond_dists = np.zeros((ind_3, ind_3))
    triplet_counts = np.zeros(ind_3, dtype=np.int8)
    for m in range(ind_3):
        pos1 = bond_positions_3[m]
        count = m+1
        trips = 0
        for n in range(m+1, ind_3):
            pos2 = bond_positions_3[n]
            diff = pos2 - pos1
            dist_curr = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2])

            if dist_curr < cutoff_3:
                cross_bond_inds[m, count] = n
                cross_bond_dists[m, count] = dist_curr
                count += 1
                trips += 1
        triplet_counts[m] = trips

    return bond_array_3, cross_bond_inds, cross_bond_dists, triplet_counts


if __name__ == '__main__':
    pass
