#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Contained Code which allows for the
"""

import numpy as np
import numpy.random as rand

from math import sin, cos
from struc import Structure

import sys
sys.path.append('../otf_engine')

from qe_util import run_espresso

def perturb_position(pos: np.array, r: float=.1, rscale = .02,):
    """

    :param pos:
    :param r:
    :param rscale:
    :return:
    """

    theta = rand.uniform(-180,180)
    x = rand.uniform(0,1)
    phi = np.arccos(2*x-1)

    R = np.abs(rand.normal(loc=r,scale=rscale))

    newpos = np.zeros(3)
    newpos[0] = pos[0] + R * sin(phi) * cos(theta)
    newpos[1] = pos[1] + R * sin(phi) * sin(theta)
    newpos[2] = pos[2] + R * cos(phi)

    return newpos


def perturb_outside_radius(structure: Structure, atom: int, r_fix: float,
                           mean_pert: float, pert_sigma = .02):


    assert r_fix > 0, "Radius must be positive"
    assert atom >= 0, "Atomic index must be non-negative"
    assert mean_pert >= 0, "Perturbation must be non-negative"


    new_positions = []
    central_pos = structure.positions[atom]

    for pos in structure.positions:
        if np.linalg.norm(pos-central_pos) > r_fix:
            new_positions.append(perturb_position(pos,mean_pert,pert_sigma))
        else:
            new_positions.append(pos)

    newstruc = Structure(lattice=structure.lattice, species=structure.species,
                         positions=new_positions,cutoff=structure.cutoff)

    return newstruc














