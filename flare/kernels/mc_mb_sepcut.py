"""Multi-element 2-, 3-, and 2+3-body kernels that restrict all signal
variance hyperparameters to a single value."""
import numpy as np
from numba import njit
from math import exp
import sys
import os
from flare.env import AtomicEnvironment
import flare.cutoffs as cf
from flare.kernels.kernels import coordination_number, q_value, q_value_mc, \
    mb_grad_helper_ls_, mb_grad_helper_ls_, k_sq_exp_double_dev, k_sq_exp_dev
from typing import Callable


@njit
def many_body_mc_sepcut_jit_(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2,
                             num_neigh_1, num_neigh_2, c1, c2, etypes1, etypes2,
                             etypes_neigh_1, etypes_neigh_2, species1, species2,
                             d1, d2, sig, ls, r_cut, cutoff_func,
                             nspec, spec_mask, mb_mask):
    """many-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        neigh_dists_2 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the second local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        num_neigh_2 (np.ndarray): number of neighbours of each atom in the second
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        etypes_neigh_2 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(
        list(set(species1).union(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    # loop over all possible species
    for s in useful_species:

        bs = spec_mask[s]
        bsn = bs * nspec
        mbtype1 = mb_mask[bc1n + bs]
        mbtype2 = mb_mask[bc2n + bs]

        t1ls = ls[mbtype1]
        t1sig = sig[mbtype1]
        t1r_cut = r_cut[mbtype1]

        t2ls = ls[mbtype2]
        t2sig = sig[mbtype2]
        t2r_cut = r_cut[mbtype2]

        # Calculate many-body descriptor values for central atoms 1 and 2
        q1 = q_value_mc(bond_array_1[:, 0], t1r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], t2r_cut, s, etypes2, cutoff_func)

        # kernel is nonzero only if central atoms are of the same species
        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, t1sig, t1ls)
        else:
            k12 = 0

        qis = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])

        qjs = np.zeros(bond_array_2.shape[0])
        qj2_grads = np.zeros(bond_array_2.shape[0])
        q2j_grads = np.zeros(bond_array_2.shape[0])
        k1js = np.zeros(bond_array_2.shape[0])

        # Loop over neighbours i of 1
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            be = spec_mask[etypes1[i]]

            if etypes1[i] == s:
                qi1, qi1_grads[i] = coordination_number(
                    ri1, ci1, t1r_cut, cutoff_func)

            if c1 == s:
                mbtype = mb_mask[bc1n + be]
                qi1, q1i_grads[i] = coordination_number(
                    ri1, ci1, r_cut[mbtype], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            # TO DO, qis[i] is 0 anyway
            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_double_dev(qis[i], q2, t2sig, t2ls)

        # Loop over neighbours j of 2
        for j in range(bond_array_2.shape[0]):
            rj2 = bond_array_2[j, 0]
            cj2 = bond_array_2[j, d2]

            be = spec_mask[etypes2[j]]
            mbtype = mb_mask[be+bsn]

            if etypes2[j] == s:
                qj2, qj2_grads[j] = coordination_number(
                    rj2, cj2, t2r_cut, cutoff_func)

            if c2 == s:
                qj2, q2j_grads[j] = coordination_number(
                    rj2, cj2, r_cut[mbtype], cutoff_func)

            # Calculate many-body descriptor value for j
            qjs[j] = q_value_mc(neigh_dists_2[j, :num_neigh_2[j]], r_cut[mbtype],
                                s, etypes_neigh_2[j, :num_neigh_2[j]], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c1 == etypes2[j]:
                k1js[j] = k_sq_exp_double_dev(q1, qjs[j], t1sig, t1ls)

        for i in range(bond_array_1.shape[0]):
            for j in range(bond_array_2.shape[0]):
                # kernel is nonzero only if central atoms are of the same species
                # TO DO
                if etypes1[i] == etypes2[j]:
                    be = spec_mask[etypes1[i]]
                    mbtype = mb_mask[be+bsn]
                    kij = k_sq_exp_double_dev(
                        qis[i], qjs[j], sig[mbtype], ls[mbtype])
                else:
                    kij = 0

                kern += qi1_grads[i] * qj2_grads[j] * \
                    (k12 + ki2s[i] + k1js[j] + kij)

    return kern


def many_body_mc_sepcut_jit(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2,
                            num_neigh_1, num_neigh_2, c1, c2,
                            etypes1, etypes2, etypes_neigh_1, etypes_neigh_2,
                            species1, species2, d1, d2, sig, ls, r_cut, cutoff_func,
                            nspec, spec_mask, mb_mask):
    """many-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        neigh_dists_2 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the second local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        num_neigh_2 (np.ndarray): number of neighbours of each atom in the second
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        etypes_neigh_2 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(
        list(set(species1).union(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    # loop over all possible species
    for s in useful_species:

        bs = spec_mask[s]
        bsn = bs * nspec
        mbtype1 = mb_mask[bc1n + bs]
        mbtype2 = mb_mask[bc2n + bs]

        t1ls = ls[mbtype1]
        t1sig = sig[mbtype1]
        t1r_cut = r_cut[mbtype1]

        t2ls = ls[mbtype2]
        t2sig = sig[mbtype2]
        t2r_cut = r_cut[mbtype2]

        # Calculate many-body descriptor values for central atoms 1 and 2
        q1 = q_value_mc(bond_array_1[:, 0],
                        r_cut[mbtype1], s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0],
                        r_cut[mbtype2], s, etypes2, cutoff_func)

        # compute kernel between central atoms only if central atoms are of the same species
        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, sig[mbtype1], ls[mbtype1])
        else:
            k12 = 0

        # initialise arrays of many body descriptors and gradients for the neighbour atoms in
        # the two configurations
        qis = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])

        qjs = np.zeros(bond_array_2.shape[0])
        qj2_grads = np.zeros(bond_array_2.shape[0])
        q2j_grads = np.zeros(bond_array_2.shape[0])
        k1js = np.zeros(bond_array_2.shape[0])

        # Loop over neighbours i of 1st configuration
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            be = spec_mask[etypes1[i]]

            if etypes1[i] == s:
                # derivative of pairwise component of many body descriptor q1i
                _, q1i_grads[i] = coordination_number(
                    ri1, ci1, r_cut[mbtype1], cutoff_func)

            if c1 == s:
                mbtype = mb_mask[bc1n + be]
                # derivative of pairwise component of many body descriptor qi1
                _, qi1_grads[i] = coordination_number(
                    ri1, ci1, r_cut[mbtype], cutoff_func)

            # Calculate many-body descriptor value for i
            mbtype = mb_mask[bsn + be]
            qis[i] = q_value_mc(neigh_dists_1[i, :num_neigh_1[i]], r_cut[mbtype],
                                s, etypes_neigh_1[i, :num_neigh_1[i]], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_double_dev(qis[i], q2, t2sig, t2ls)

        # Loop over neighbours j of 2
        for j in range(bond_array_2.shape[0]):
            rj2 = bond_array_2[j, 0]
            cj2 = bond_array_2[j, d2]

            be = spec_mask[etypes2[j]]
            mbtype = mb_mask[be+bsn]

            if etypes2[j] == s:
                _, q2j_grads[j] = coordination_number(
                    rj2, cj2, t2r_cut, cutoff_func)

            if c2 == s:
                _, qj2_grads[j] = coordination_number(
                    rj2, cj2, r_cut[mbtype], cutoff_func)

            # Calculate many-body descriptor value for j
            qjs[j] = q_value_mc(neigh_dists_2[j, :num_neigh_2[j]], r_cut[mbtype],
                                s, etypes_neigh_2[j, :num_neigh_2[j]], cutoff_func)

            # kernel is nonzero only if central atoms are of the same species
            if c1 == etypes2[j]:
                k1js[j] = k_sq_exp_double_dev(q1, qjs[j], t1sig, t1ls)

        for i in range(bond_array_1.shape[0]):
            for j in range(bond_array_2.shape[0]):
                # kernel is nonzero only if central atoms are of the same species
                if etypes1[i] == etypes2[j]:
                    be = spec_mask[etypes1[i]]
                    mbtype = mb_mask[be+bsn]
                    kij = k_sq_exp_double_dev(
                        qis[i], qjs[j], sig[mbtype], ls[mbtype])
                else:
                    kij = 0

                kern += q1i_grads[i] * q2j_grads[j] * k12
                kern += qi1_grads[i] * q2j_grads[j] * ki2s[i]
                kern += q1i_grads[i] * qj2_grads[j] * k1js[j]
                kern += qi1_grads[i] * qj2_grads[j] * kij
    return kern


@njit
def many_body_mc_grad_sepcut_jit(bond_array_1, bond_array_2, neigh_dists_1, neigh_dists_2, num_neigh_1,
                                 num_neigh_2, c1, c2, etypes1, etypes2, etypes_neigh_1, etypes_neigh_2,
                                 species1, species2, d1, d2, sig, ls, r_cut, cutoff_func,
                                 nspec, spec_mask, nmb, mb_mask):
    """gradient of many-body multi-element kernel between two force components
    w.r.t. the hyperparameters, accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        neigh_dists_2 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the second local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        num_neigh_2 (np.ndarray): number of neighbours of each atom in the second
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        etypes_neigh_2 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        array: Value of the many-body kernel and its gradient w.r.t. sig and ls
    """

    kern = 0
    sig_derv = np.zeros(nmb)
    ls_derv = np.zeros(nmb)

    useful_species = np.array(
        list(set(species1).union(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    for s in useful_species:

        bs = spec_mask[s]
        bsn = bs * nspec
        mbtype1 = mb_mask[bc1n + bs]
        mbtype2 = mb_mask[bc2n + bs]

        t1ls = ls[mbtype1]
        t1sig = sig[mbtype1]
        t1r_cut = r_cut[mbtype1]

        t2ls = ls[mbtype2]
        t2sig = sig[mbtype2]
        t2r_cut = r_cut[mbtype2]

        # Calculate many-body descriptor values for 1 and 2
        q1 = q_value_mc(bond_array_1[:, 0], t1r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], t2r_cut, s, etypes2, cutoff_func)

        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, t1sig, t1ls)
            q12diffsq = (q1 - q2) ** 2  # * (q1 - q2)
            dk12 = mb_grad_helper_ls_(q12diffsq, t1sig, t1ls)
        else:
            k12 = 0
            dk12 = 0

        qis = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])
        dki2s = np.zeros(bond_array_1.shape[0])

        qjs = np.zeros(bond_array_2.shape[0])
        qj2_grads = np.zeros(bond_array_2.shape[0])
        q2j_grads = np.zeros(bond_array_2.shape[0])
        k1js = np.zeros(bond_array_2.shape[0])
        dk1js = np.zeros(bond_array_2.shape[0])

        # Compute  ki2s, qi1_grads, and qis
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            be = spec_mask[etypes1[i]]
            mbtype = mb_mask[bsn + be]

            if etypes1[i] == s:
                _, q1i_grads[i] = coordination_number(
                    ri1, ci1, t1r_cut, cutoff_func)

            if c1 == s:
                # derivative of pairwise component of many body descriptor qi1
                __, qi1_grads[i] = coordination_number(
                    ri1, ci1, r_cut[mbtype], cutoff_func)

            # Calculate many-body descriptor value for i
            qis[i] = q_value_mc(neigh_dists_1[i, :num_neigh_1[i]], r_cut[mbtype],
                                s, etypes_neigh_1[i, :num_neigh_1[i]], cutoff_func)

            # ki2s[i] = k_sq_exp_double_dev(qis[i], q2, sig, ls)
            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_double_dev(qis[i], q2, t2sig, t2ls)
                qi2diffsq = (qis[i] - q2) * (qis[i] - q2)
                dki2s[i] = mb_grad_helper_ls_(qi2diffsq, t2sig, t2ls)

        # Compute k1js, qj2_grads and qjs
        for j in range(bond_array_2.shape[0]):
            rj2 = bond_array_2[j, 0]
            cj2 = bond_array_2[j, d2]

            be = spec_mask[etypes2[j]]
            mbtype = mb_mask[bsn + be]
            tr_cut = r_cut[mbtype]

            if etypes2[j] == s:
                _, q2j_grads[j] = coordination_number(
                    rj2, cj2, t2r_cut, cutoff_func)

            if c2 == s:
                _, qj2_grads[j] = coordination_number(
                    rj2, cj2, tr_cut, cutoff_func)

            # Calculate many-body descriptor value for j
            qjs[j] = q_value_mc(neigh_dists_2[j, :num_neigh_2[j]], tr_cut,
                                s, etypes_neigh_2[j, :num_neigh_2[j]], cutoff_func)

            # k1js[j] = k_sq_exp_double_dev(q1, qjs[j], sig, ls)

            if c1 == etypes2[j]:
                k1js[j] = k_sq_exp_double_dev(q1, qjs[j], t1sig, t1ls)
                q1jdiffsq = (q1 - qjs[j]) * (q1 - qjs[j])
                dk1js[j] = mb_grad_helper_ls_(q1jdiffsq, t1sig, t1ls)

        for i in range(bond_array_1.shape[0]):
            for j in range(bond_array_2.shape[0]):

                # kij = k_sq_exp_double_dev(qis[i], qjs[j], sig, ls)
                be = spec_mask[etypes2[j]]
                mbtype = mb_mask[bsn + be]

                if etypes1[i] == etypes2[j]:
                    kij = k_sq_exp_double_dev(
                        qis[i], qjs[j], sig[mbtype], ls[mbtype])
                    qijdiffsq = (qis[i] - qjs[j]) * (qis[i] - qjs[j])
                    dkij = mb_grad_helper_ls_(
                        qijdiffsq, sig[mbtype], ls[mbtype])
                else:
                    kij = 0
                    dkij = 0

                # c1 s and c2 s and if c1==c2 --> c1 s
                kern_term_c1s = q1i_grads[i] * q2j_grads[j] * k12
                sig_derv[mbtype1] += kern_term_c1s * 2. / t1sig
                kern += kern_term_c1s

                # s e1 and c2 s and c2==e1 --> c2 s
                kern_term_c2s = qi1_grads[i] * q2j_grads[j] * ki2s[i]
                sig_derv[mbtype2] += kern_term_c2s * 2. / t2sig
                kern += kern_term_c2s

                # c1 s and s e2 and  c1==e2 --> c1 s
                kern_term_c1s = q1i_grads[i] * qj2_grads[j] * k1js[j]
                sig_derv[mbtype1] += kern_term_c1s * 2. / t1sig
                kern += kern_term_c1s

                # s e1 and s e2 and e1 == e2 -> s e
                kern_term_se = qi1_grads[i] * qj2_grads[j] * kij
                sig_derv[mbtype] += kern_term_se * 2. / sig[mbtype]
                kern += kern_term_se

                ls_derv[mbtype1] += q1i_grads[i] * q2j_grads[j] * dk12
                ls_derv[mbtype2] += qi1_grads[i] * q2j_grads[j] * dki2s[i]
                ls_derv[mbtype1] += q1i_grads[i] * qj2_grads[j] * dk1js[j]
                ls_derv[mbtype] += qi1_grads[i] * qj2_grads[j] * dkij

    grad = np.zeros(nmb*2)
    grad[:nmb] = sig_derv
    grad[nmb:] = ls_derv

    return kern, grad


@njit
def many_body_mc_force_en_sepcut_jit(bond_array_1, bond_array_2, neigh_dists_1, num_neigh_1,
                                     c1, c2, etypes1, etypes2, etypes_neigh_1,
                                     species1, species2, d1, sig, ls, r_cut, cutoff_func,
                                     nspec, spec_mask, mb_mask):
    """many-body many-element kernel between force and energy components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        neigh_dists_1 (np.ndarray): matrix padded with zero values of distances
            of neighbours for the atoms in the first local environment.
        num_neigh_1 (np.ndarray): number of neighbours of each atom in the first
            local environment
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        etypes_neigh_1 (np.ndarray): atomic species of atoms in the neighbourhoods
            of atoms in env 1
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(
        list(set(species1).union(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    for s in useful_species:

        bs = spec_mask[s]
        bsn = bs * nspec
        mbtype1 = mb_mask[bc1n + bs]
        mbtype2 = mb_mask[bc2n + bs]

        t1ls = ls[mbtype1]
        t1sig = sig[mbtype1]
        t1r_cut = r_cut[mbtype1]

        t2ls = ls[mbtype2]
        t2sig = sig[mbtype2]
        t2r_cut = r_cut[mbtype2]

        q1 = q_value_mc(bond_array_1[:, 0], t1r_cut, s, etypes1, cutoff_func)
        q2 = q_value_mc(bond_array_2[:, 0], t2r_cut, s, etypes2, cutoff_func)

        if c1 == c2:
            k12 = k_sq_exp_dev(q1, q2, t1sig, t1ls)
        else:
            k12 = 0

        qis = np.zeros(bond_array_1.shape[0])
        qi1_grads = np.zeros(bond_array_1.shape[0])
        q1i_grads = np.zeros(bond_array_1.shape[0])
        ki2s = np.zeros(bond_array_1.shape[0])

        # Loop over neighbours i of 1
        for i in range(bond_array_1.shape[0]):
            ri1 = bond_array_1[i, 0]
            ci1 = bond_array_1[i, d1]

            be = spec_mask[etypes1[i]]
            mbtype = mb_mask[bsn + be]

            if etypes1[i] == s:
                _, q1i_grads[i] = coordination_number(
                    ri1, ci1, t1r_cut, cutoff_func)

            if c1 == s:
                _, qi1_grads[i] = coordination_number(
                    ri1, ci1, r_cut[mbtype], cutoff_func)

            # Calculate many-body descriptor value for i
            qis[i] = q_value_mc(neigh_dists_1[i, :num_neigh_1[i]], r_cut[mbtype],
                                s, etypes_neigh_1[i, :num_neigh_1[i]], cutoff_func)

            if c2 == etypes1[i]:
                ki2s[i] = k_sq_exp_dev(qis[i], q2, t2sig, t2ls)

            kern += - (q1i_grads[i] * k12 + qi1_grads[i] * ki2s[i])

    return kern


@njit
def many_body_mc_en_sepcut_jit(bond_array_1, bond_array_2, c1, c2, etypes1, etypes2,
                               species1, species2,
                               sig, ls, r_cut, cutoff_func,
                               nspec, spec_mask, mb_mask):
    """many-body many-element kernel between energy components accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): many-body bond array of the first local
            environment.
        bond_array_2 (np.ndarray): many-body bond array of the second local
            environment.
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.
        r_cut (float): many-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Return:
        float: Value of the many-body kernel.
    """
    useful_species = np.array(
        list(set(species1).union(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    kern = 0

    ls2 = ls*ls
    sig2 = sig*sig

    if c1 == c2:

        for s in useful_species:

            bs = spec_mask[s]
            mbtype = mb_mask[bc1n + bs]

            tls2 = ls2[mbtype]
            tsig2 = sig2[mbtype]
            tr_cut = r_cut[mbtype]

            q1 = q_value_mc(bond_array_1[:, 0],
                            tr_cut, s, etypes1, cutoff_func)
            q2 = q_value_mc(bond_array_2[:, 0],
                            tr_cut, s, etypes2, cutoff_func)
            q1q2diff = q1 - q2

            kern += tsig2 * exp(-q1q2diff * q1q2diff / (2 * tls2))

    return kern
