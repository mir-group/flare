#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

""""
Test suite for the rcut.py file, which systematically determines a good cutoff
radius

Steven Torrisi
"""

import pytest
import os
from numpy import arccos, zeros, arctan, isclose, pi, equal, not_equal, diag, \
    ones
from numpy.linalg import norm
from numpy.random import randint, normal, uniform

import sys

sys.path.append('../otf_engine')
sys.path.append('../modules')
from struc import Structure
from numpy import eye
from rcut import perturb_position, perturb_outside_radius, \
    is_within_r_periodic, gauge_force_variance
from test_qe_util import cleanup_espresso_run


def test_perturb():
    """
    Tests to make sure the randomly distributed positions are actually
    uniformly distributed over the surface of a sphere of desired avg radius
    """
    trials = 200000

    running_phi = 0
    running_theta = 0
    running_radius = 0
    for n in range(trials):
        pos = zeros(3)
        newpos = perturb_position(pos, r_pert=1, rscale=.1)

        rad = norm(newpos)

        running_theta += arccos(newpos[2] / rad)
        running_phi += arctan(newpos[1] / newpos[0])
        running_radius += rad

    avg_phi = running_phi / trials
    avg_theta = running_theta / trials
    avg_rad = running_radius / trials
    assert isclose(avg_phi, 0, atol=.2)
    assert isclose(avg_theta, pi / 2, atol=.2)
    assert isclose(avg_rad, 1, atol=.2)


def test_perturb_outside_radius():
    # Try for 10 different random structures

    for _ in range(10):
        d = abs(normal(10, 2, 3))
        cell = diag(d)
        noa = randint(5, 30)
        positions = [uniform(0, d, 3) for _ in range(noa)]

        struc = Structure(lattice=cell, species=['A'] * noa,
                          positions=positions,
                          cutoff=1)
        target_index = randint(0, noa)
        target_pos = struc.positions[target_index]
        r_fix = abs(normal(5.0, 1.0))
        within_radius = [is_within_r_periodic(struc, target_pos, pos, r_fix)
                         for
                         pos in
                         struc.positions]

        pert_struc = perturb_outside_radius(struc, target_index, r_fix=r_fix,
                                            mean_pert=1.,pert_sigma=.02)

        for i, pert_pos in enumerate(pert_struc.positions):
            if within_radius[i]:
                assert equal(pert_pos, struc.positions[i]).all()
            else:
                assert not_equal(pert_pos, struc.positions[i]).all()


def test_within_radius():
    struc = Structure(lattice=eye(3), positions=[zeros(3), .9 * ones(3)],
                      species=[], cutoff=1)

    assert is_within_r_periodic(struc, zeros(3), .9 * ones(3), radius=.2)

    images = struc.get_periodic_images(.9 * ones(3))

    dists = [norm(image) for image in images]

    assert min(dists) == 0.17320508075688767

    pert_struc_1 = perturb_outside_radius(struc, 0, r_fix=.05,
                                          mean_pert=1.,pert_sigma=.02)

    assert equal(struc.positions[0], pert_struc_1.positions[0]).all()
    assert not equal(struc.positions[1], pert_struc_1.positions[1]).all()

    pert_struc_2 = perturb_outside_radius(struc, 0, r_fix=.174,
                                          mean_pert=1.,pert_sigma=.02)

    assert equal(struc.positions[0], pert_struc_2.positions[0]).all()
    assert equal(struc.positions[1], pert_struc_2.positions[1]).all()


def test_gauge_rcut_run():
    """
    Runs an rcut run on the H2 dimer; checks that the forces did not change
    for the first 'perturbation' (in which the second atom does not move)
    and that the forces do change for the second perturbation.
    :return:
    """
    os.system('cp ./test_files/qe_input_1.in pwscf.in')

    total_forces_1 = gauge_force_variance(qe_input='pwscf.in', trials=3,
                                          atom=0, r_fix=3.0,
                                          mean_pert=.1,write_output=True)

    for i in range(3):
        for j in range(3):
            if i != j:
                assert isclose(total_forces_1[i], total_forces_1[j]).all()

    total_forces_2 = gauge_force_variance(qe_input='pwscf.in', trials=3,
                                          atom=0, r_fix=.5, mean_pert=.1,
                                          write_output=True)

    for i in range(3):
        for j in range(3):
            if i != j:
                assert not isclose(total_forces_2[i], total_forces_2[j]).all()

    os.system('rm rcut.out')
    cleanup_espresso_run()
