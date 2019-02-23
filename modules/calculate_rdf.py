import numpy as np
import sys
sys.path.append('../../otf/otf_engine')
import struc, env


def calculate_rdf(position_list, cell, species, snaps, cutoff, bins):
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
    cell_vol = cell[0, 0]**3
    rho = nat / cell_vol
    weights = (4 * np.pi * rho / 3) * ((rs+delta_r)**3 - rs**3)
    rad_dist = radial_hist / (atom_count * weights)

    return rs, rad_dist, atom_count
