#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Contained Code which allows for the
"""

import sys

import numpy as np
import numpy.random as rand

sys.path.append('../otf_engine')

from math import sin, cos
from struc import Structure
from qe_util import run_espresso, parse_qe_input


def perturb_position(pos, r_pert: float, rscale):
    """

    :param pos:
    :param r_pert:
    :param rscale:
    :return:
    """

    theta = rand.uniform(-180, 180)
    x = rand.uniform(0, 1)
    phi = np.arccos(2 * x - 1)

    r = np.abs(rand.normal(loc=r_pert, scale=rscale))

    newpos = np.zeros(3)
    newpos[0] = pos[0] + r * sin(phi) * cos(theta)
    newpos[1] = pos[1] + r * sin(phi) * sin(theta)
    newpos[2] = pos[2] + r * cos(phi)

    return newpos


def perturb_outside_radius(structure: Structure, atom: int, r_fix: float,
                           mean_pert: float, pert_sigma: float):
    """

    :param structure:
    :param atom:
    :param r_fix:
    :param mean_pert:
    :param pert_sigma:
    :return:
    """
    assert r_fix > 0, "Radius must be positive"
    assert atom >= 0, "Atomic index must be non-negative"
    assert mean_pert >= 0, "Perturbation must be non-negative"

    new_positions = []
    central_pos = structure.positions[atom]

    for pos in structure.positions:
        if is_within_r_periodic(structure, central_pos, pos, r_fix):
            new_positions.append(pos)
        else:
            new_positions.append(perturb_position(pos, mean_pert, pert_sigma))

    newstruc = Structure(lattice=structure.lattice, species=structure.species,
                         positions=new_positions, cutoff=structure.cutoff)

    return newstruc


def is_within_r_periodic(structure, central_pos, neighbor_pos, radius):
    """

    :param structure:
    :param central_pos:
    :param neighbor_pos:
    :param radius:
    :return:
    """
    images = structure.get_periodic_images(neighbor_pos, super_check=2)
    # Check to see if images are within the radius
    for image in images:
        if np.linalg.norm(image - central_pos) < radius:
            return True

    return False


def gauge_force_variance(qe_input: str, trials: int, atom: int, r_fix:
    float, mean_pert: float = .1, pert_sigma: float = .02, write_output:
    bool = True):
    """

    :param qe_input:
    :param trials:
    :param atom:
    :param r_fix:
    :param mean_pert:
    :param pert_sigma:
    :param write_output:
    :return:
    """
    positions, species, cell, masses = parse_qe_input(qe_input)
    struc = Structure(positions=positions, species=species, lattice=cell,
                      cutoff=1, mass_dict=masses)

    total_forces = np.empty(shape=(trials, struc.nat, 3))

    for i in range(trials):
        pert_struct = perturb_outside_radius(struc, atom, r_fix, mean_pert,
                                             pert_sigma)
        pert_forces = run_espresso(qe_input=qe_input, structure=pert_struct,
                                   temp=True)

        for j in range(struc.nat):
            for k in range(3):
                total_forces[i, j, k] = float(pert_forces[j][k])

        if write_output:
            with open("rcut.out", 'a') as f:
                f.write('Perturbed structure about atom {} and radius {}'
                        ': \n'.format(atom, r_fix))
                for pos in pert_struct.positions:
                    f.write(str(pos)+'\n')
                f.write("All Forces (normed force on atom {}: {}) : "
                        "\n".format(atom,
                                    np.linalg.norm(total_forces[i,atom,:])))
                for force in total_forces[i,:]:
                    f.write(str(force)+'\n')
                #f.write(str(total_forces[i, :, :]))

    all_forces = [total_forces[i][atom][0:3] for i in range(trials)]

    report_str = 'Std dev of force on atom {} : {} Ry/Au \n'.format(atom,
        np.std([np.linalg.norm(force) for force in all_forces]))

    print(report_str)

    if write_output:
        with open("rcut.out", 'a') as f:
            f.write(report_str)

    return total_forces


if __name__ == '__main__':
    # gauge_force_variance(qe_input='qe_input_1.in', trials=5, atom=0,
    # r_fix=3.0,
    #                     mean_pert=.1)
    pass
