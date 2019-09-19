import numpy as np
from math import cos, pi, exp


def cutoff_func(rij, rc):
    if rij < rc:
        return 0.5 * (cos(pi * rij / rc) + 1)
    else:
        return 0


def exp_func(rij, rs, eta=1):
    return exp(-eta * (rij-rs)**2)


def behler_radial(rij, rs, rc, eta=1):
    return cutoff_func(rij, rc) * exp_func(rij, rs, eta)


if __name__ == '__main__':
    from flare import env, struc

    cell = np.eye(3) * 10
    species = [0, 1, 1]
    positions = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0]])
    cutoffs = np.array([2., 2.])

    structure = struc.Structure(cell, species, positions)
    environment = env.AtomicEnvironment(structure, 0, cutoffs,
                                        compute_angles=True)

    print(environment.cos_thetas)