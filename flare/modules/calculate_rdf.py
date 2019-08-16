import numpy as np
import sys
from flare import struc, env


def calculate_rdf(position_list, cell, species, snaps, cutoff, bins,
                  cell_vol=None):

    # assume cubic cell by default
    if cell_vol is None:
        cell_vol = cell[0, 0] ** 3

    # collect interatomic distances
    r_list = []
    delta_r = cutoff / bins
    atom_count = 0
    nat = position_list[0].shape[0]
    cutoffs = np.array([cutoff])
    for snap in snaps:
        positions = position_list[snap]
        structure = struc.Structure(cell, species, positions)

        for n in range(len(positions)):
            env_curr = env.AtomicEnvironment(structure, n, cutoffs)
            atom_count += 1
            for bond in env_curr.bond_array_2:
                r_list.append(bond[0])
    r_list = np.array(r_list)
    radial_hist, _ = \
        np.histogram(r_list, bins=bins, range=(0, cutoff))

    # weight the histogram
    rs = np.linspace(delta_r/2, cutoff-delta_r/2, bins)
    rho = nat / cell_vol
    weights = (4 * np.pi * rho / 3) * ((rs+delta_r)**3 - rs**3)
    rad_dist = radial_hist / (atom_count * weights)

    return rs, rad_dist, atom_count


def calculate_species_rdf(position_list, spec1, spec2, cell, species, snaps,
                          cutoff, bins, cell_vol=None):

    # assume cubic cell by default
    if cell_vol is None:
        cell_vol = cell[0, 0] ** 3

    # collect interatomic distances
    r_list = []
    delta_r = cutoff / bins
    atom_count = 0
    nat = position_list[0].shape[0]
    cutoffs = np.array([cutoff])

    # compute concentration of species 2
    positions = position_list[snaps[0]]
    struc_ex = struc.Structure(cell, species, positions)
    spec2_count = 0
    for spec in struc_ex.coded_species:
        if spec == spec2:
            spec2_count += 1
    spec2_conc = spec2_count / nat

    for snap in snaps:
        positions = position_list[snap]
        structure = struc.Structure(cell, species, positions)

        for n in range(len(positions)):
            env_curr = env.AtomicEnvironment(structure, n, cutoffs)
            ctype = env_curr.ctype

            if ctype == spec1:
                atom_count += 1

                for bond, spec in zip(env_curr.bond_array_2, env_curr.etypes):
                    if spec == spec2:
                        r_list.append(bond[0])

    r_list = np.array(r_list)
    radial_hist, _ = \
        np.histogram(r_list, bins=bins, range=(0, cutoff))

    # weight the histogram
    rs = np.linspace(delta_r/2, cutoff-delta_r/2, bins)
    rho = (nat * spec2_conc) / cell_vol
    weights = (4 * np.pi * rho / 3) * ((rs+delta_r)**3 - rs**3)
    rad_dist = radial_hist / (atom_count * weights)

    return rs, rad_dist, atom_count
