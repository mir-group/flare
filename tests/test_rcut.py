#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

""""
Test suite for the rcut.py file, which systematically determines a good cutoff
radius

Steven Torrisi
"""

import pytest

from numpy import arccos, zeros, arctan,isclose,pi,eye,equal,not_equal
from numpy.linalg import norm
from numpy.random import randint,normal,uniform
from test_GaussianProcess import get_random_structure

import sys
sys.path.append('../otf_engine')
from struc import Structure
from qe_util import run_espresso
from rcut import perturb_position,perturb_outside_radius


def test_perturb():
    """
    Tests to make sure the randomly distributed positions are actually
    uniformly distributed over the surface of a sphere of desired avg radius
    """
    trials = 200000

    running_phi    = 0
    running_theta  = 0
    running_radius = 0
    for n in range(trials):
        pos = zeros(3)
        newpos = perturb_position(pos,r=1,rscale=.1)

        rad = norm(newpos)

        running_theta += arccos(newpos[2]/rad)
        running_phi   += arctan(newpos[1]/newpos[0])
        running_radius+= rad

    avg_phi =running_phi/trials
    avg_theta =running_theta/trials
    avg_rad = running_radius/trials
    assert isclose(avg_phi,0,atol= .2)
    assert isclose(avg_theta,pi/2,atol=.2)
    assert isclose(avg_rad,1,atol=.2)


def test_perturb_outside_radius():

    # Try for 10 different random structures

    for _ in range(10):
        d = abs(normal(10,2))
        cell = d * eye(3)
        noa = randint(5,30)
        positions = [uniform(0,d,3) for _ in range(noa)]

        struc = Structure(lattice=cell,species=['A']*noa,positions=positions,
                                        cutoff=1)
        target_index = randint(0, noa)
        target_pos = struc.positions[target_index]
        r_fix = abs(normal(5.0,1.0))
        within_radius = [norm(target_pos-pos)<r_fix for pos in struc.positions]

        pert_struc = perturb_outside_radius(struc,target_index,r_fix=r_fix,
                                            mean_pert=1.)

        for i, pert_pos in enumerate(pert_struc.positions):
            if within_radius[i]:
                assert equal(pert_pos,struc.positions[i]).all()
            else:
                assert not_equal(pert_pos,struc.positions[i]).all()




