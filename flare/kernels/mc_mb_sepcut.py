"""
Implementation of three-body kernels using different cutoffs.

The kernels are slightly slower.
"""

import numpy as np
import os
import sys

from numba import njit
from math import exp

import flare.kernels.cutoffs as cf

from flare.env import AtomicEnvironment
from flare.kernels.kernels import coordination_number, q_value, q_value_mc, \
    mb_grad_helper_ls_, mb_grad_helper_ls_, k_sq_exp_double_dev, k_sq_exp_dev
from typing import Callable


@njit
def many_body_mc_sepcut_jit(q_array_1, q_array_2,
                            q_neigh_array_1, q_neigh_array_2,
                            q_neigh_grads_1, q_neigh_grads_2,
                            c1, c2, etypes1, etypes2,
                            species1, species2,
                            d1, d2, sig, ls,
                            nspec, spec_mask, mb_mask):
    """many-body multi-element kernel between two force components accelerated
    with Numba.

    Args:
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8)

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

        # Calculate many-body descriptor values for central atoms 1 and 2
        s1 = np.where(species1==s)[0][0]
        s2 = np.where(species2==s)[0][0]
        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        # compute kernel only if central atoms are of the same species
        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, sig[mbtype1], ls[mbtype1])
        else:
            k12 = 0

        # initialise arrays of many body descriptors and gradients for the neighbour atoms in
        # the two configurations
        # Loop over neighbours i of 1st configuration
        for i in range(q_neigh_array_1.shape[0]):
            qis = q1i_grads = qi1_grads = ki2s = 0
            if etypes1[i] == s:
                q1i_grads = q_neigh_grads_1[i, d1-1]

            if c1 == s:
                qi1_grads = q_neigh_grads_1[i, d1-1]

            # Calculate many-body descriptor value for i
            qis = q_neigh_array_1[i, s1]

            if c2 == etypes1[i]:
                ki2s = k_sq_exp_double_dev(qis, q2, sig[mbtype2], ls[mbtype2])

            # Loop over neighbours j of 2
            for j in range(q_neigh_array_2.shape[0]):
                qjs = qj2_grads = q2j_grads = k1js = 0

                if etypes2[j] == s:
                    q2j_grads = q_neigh_grads_2[j, d2-1]

                if c2 == s:
                    qj2_grads = q_neigh_grads_2[j, d2-1]

                # Calculate many-body descriptor value for j
                qjs = q_neigh_array_2[j, s2]

                if c1 == etypes2[j]:
                    k1js = k_sq_exp_double_dev(q1, qjs, sig[mbtype1], ls[mbtype1])

                be = spec_mask[etypes1[i]]
                mbtype = mb_mask[be+bsn]
                if etypes1[i] == etypes2[j]:
                    kij = k_sq_exp_double_dev(qis, qjs, sig[mbtype], ls[mbtype])
                else:
                    kij = 0

                kern += q1i_grads * q2j_grads * k12
                kern += qi1_grads * q2j_grads * ki2s
                kern += q1i_grads * qj2_grads * k1js
                kern += qi1_grads * qj2_grads * kij
    return kern

@njit
def many_body_mc_grad_sepcut_jit(q_array_1, q_array_2,
                            q_neigh_array_1, q_neigh_array_2,
                            q_neigh_grads_1, q_neigh_grads_2,
                            c1, c2, etypes1, etypes2,
                            species1, species2,
                            d1, d2, sig, ls,
                            nspec, spec_mask, nmb, mb_mask):
    """gradient of many-body multi-element kernel between two force components
    w.r.t. the hyperparameters, accelerated with Numba.

    Args:
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        etypes2 (np.ndarray): atomic species of atoms in env 2
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        d2 (int): Force component of the second environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.

    Return:
        array: Value of the many-body kernel and its gradient w.r.t. sig and ls
    """

    kern = 0
    sig_derv = np.zeros(nmb, dtype=np.float64)
    ls_derv = np.zeros(nmb, dtype=np.float64)

    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    for s in useful_species:

        bs = spec_mask[s]
        bsn = bs * nspec
        mbtype1 = mb_mask[bc1n + bs]
        mbtype2 = mb_mask[bc2n + bs]

        # Calculate many-body descriptor values for central atoms 1 and 2
        s1 = np.where(species1==s)[0][0]
        s2 = np.where(species2==s)[0][0]
        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        # compute kernel only if central atoms are of the same species
        if c1 == c2:
            k12 = k_sq_exp_double_dev(q1, q2, sig[mbtype1], ls[mbtype1])
            q12diffsq = (q1 - q2) ** 2  # * (q1 - q2)
            dk12 = mb_grad_helper_ls_(q12diffsq, sig[mbtype1], ls[mbtype1])
        else:
            k12 = 0
            dk12 = 0

        # Compute  ki2s, qi1_grads, and qis
        for i in range(q_neigh_array_1.shape[0]):
            qis = q1i_grads = qi1_grads = ki2s = dki2s = 0
            if etypes1[i] == s:
                q1i_grads = q_neigh_grads_1[i, d1-1]

            if c1 == s:
                qi1_grads = q_neigh_grads_1[i, d1-1]

            # Calculate many-body descriptor value for i
            qis = q_neigh_array_1[i, s1]

            if c2 == etypes1[i]:
                ki2s = k_sq_exp_double_dev(qis, q2, sig[mbtype2], ls[mbtype2])
                qi2diffsq = (qis - q2) * (qis - q2)
                dki2s = mb_grad_helper_ls_(qi2diffsq, sig[mbtype2], ls[mbtype2])

            # Compute k1js, qj2_grads and qjs
            for j in range(q_neigh_array_2.shape[0]):
                qjs = qj2_grads = q2j_grads = k1js = dk1js = 0
                if etypes2[j] == s:
                    q2j_grads = q_neigh_grads_2[j, d2-1]

                if c2 == s:
                    qj2_grads = q_neigh_grads_2[j, d2-1]

                # Calculate many-body descriptor value for j
                qjs = q_neigh_array_2[j, s2]

                if c1 == etypes2[j]:
                    k1js = k_sq_exp_double_dev(q1, qjs, sig[mbtype1], ls[mbtype1])
                    q1jdiffsq = (q1 - qjs) * (q1 - qjs)
                    dk1js = mb_grad_helper_ls_(q1jdiffsq, sig[mbtype1], ls[mbtype1])

                be = spec_mask[etypes2[j]]
                mbtype = mb_mask[bsn + be]
                if etypes1[i] == etypes2[j]:
                    kij = k_sq_exp_double_dev(
                        qis, qjs, sig[mbtype], ls[mbtype])
                    qijdiffsq = (qis - qjs) * (qis - qjs)
                    dkij = mb_grad_helper_ls_(
                        qijdiffsq, sig[mbtype], ls[mbtype])
                else:
                    kij = 0
                    dkij = 0

                # c1 s and c2 s and if c1==c2 --> c1 s
                if k12 != 0:
                    kern_term_c1s = q1i_grads * q2j_grads * k12
                    if sig[mbtype1] !=0:
                        sig_derv[mbtype1] += kern_term_c1s * 2. / sig[mbtype1]
                    kern += kern_term_c1s
                    ls_derv[mbtype1] += q1i_grads * q2j_grads * dk12

                # s e1 and c2 s and c2==e1 --> c2 s
                if ki2s != 0:
                    kern_term_c2s = qi1_grads * q2j_grads * ki2s
                    if sig[mbtype2] !=0:
                        sig_derv[mbtype2] += kern_term_c2s * 2. / sig[mbtype2]
                    kern += kern_term_c2s
                    ls_derv[mbtype2] += qi1_grads * q2j_grads * dki2s

                # c1 s and s e2 and  c1==e2 --> c1 s
                if k1js != 0:
                    kern_term_c1s = q1i_grads * qj2_grads * k1js
                    if sig[mbtype1] !=0:
                        sig_derv[mbtype1] += kern_term_c1s * 2. / sig[mbtype1]
                    kern += kern_term_c1s
                    ls_derv[mbtype1] += q1i_grads * qj2_grads * dk1js

                # s e1 and s e2 and e1 == e2 -> s e
                if kij != 0:
                    kern_term_se = qi1_grads * qj2_grads * kij
                    if sig[mbtype] !=0:
                        sig_derv[mbtype] += kern_term_se * 2. / sig[mbtype]
                    kern += kern_term_se
                    ls_derv[mbtype]  += qi1_grads * qj2_grads * dkij


    grad = np.zeros(nmb*2, dtype=np.float64)
    grad[:nmb] = sig_derv
    grad[nmb:] = ls_derv

    return kern, grad


@njit
def many_body_mc_force_en_sepcut_jit(q_array_1, q_array_2,
                                     q_neigh_array_1, q_neigh_grads_1,
                                     c1, c2, etypes1,
                                     species1, species2, d1, sig, ls,
                                     nspec, spec_mask, mb_mask):
    """many-body many-element kernel between force and energy components accelerated
    with Numba.

    Args:
        To be complete
        c1 (int): atomic species of the central atom in env 1
        c2 (int): atomic species of the central atom in env 2
        etypes1 (np.ndarray): atomic species of atoms in env 1
        species1 (np.ndarray): all the atomic species present in trajectory 1
        species2 (np.ndarray): all the atomic species present in trajectory 2
        d1 (int): Force component of the first environment.
        sig (float): many-body signal variance hyperparameter.
        ls (float): many-body length scale hyperparameter.

    Return:
        float: Value of the many-body kernel.
    """

    kern = 0

    useful_species = np.array(
        list(set(species1).intersection(set(species2))), dtype=np.int8)

    bc1 = spec_mask[c1]
    bc1n = bc1 * nspec
    bc2 = spec_mask[c2]
    bc2n = bc2 * nspec

    for s in useful_species:

        bs = spec_mask[s]
        bsn = bs * nspec
        mbtype1 = mb_mask[bc1n + bs]
        mbtype2 = mb_mask[bc2n + bs]

        s1 = np.where(species1==s)[0][0]
        s2 = np.where(species2==s)[0][0]
        q1 = q_array_1[s1]
        q2 = q_array_2[s2]

        if c1 == c2:
            k12 = k_sq_exp_dev(q1, q2, sig[mbtype1], ls[mbtype1])
        else:
            k12 = 0

        # Loop over neighbours i of 1
        for i in range(q_neigh_array_1.shape[0]):
            qi1_grads = q1i_grads = 0
            ki2s = 0

            if etypes1[i] == s:
                q1i_grads = q_neigh_grads_1[i, d1-1]

            if (c1 == s) and (c2 == etypes1[i]):
                qi1_grads = q_neigh_grads_1[i, d1-1]
                qis = q_neigh_array_1[i, s1]
                ki2s = k_sq_exp_dev(qis, q2, sig[mbtype2], ls[mbtype2])

            kern -= q1i_grads * k12 + qi1_grads * ki2s
    return kern


@njit
def many_body_mc_en_sepcut_jit(q_array_1, q_array_2, c1, c2,
                               species1, species2,
                               sig, ls,
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
        list(set(species1).intersection(set(species2))), dtype=np.int8)

    kern = 0

    if c1 == c2:
        ls2 = ls*ls
        sig2 = sig*sig

        bc1 = spec_mask[c1]
        bc1n = bc1 * nspec

        for s in useful_species:

            bs = spec_mask[s]
            mbtype = mb_mask[bc1n + bs]

            tls2 = ls2[mbtype]
            tsig2 = sig2[mbtype]

            q1 = q_array_1[np.where(species1==s)[0][0]]
            q2 = q_array_2[np.where(species2==s)[0][0]]

            q1q2diff = q1 - q2

            kern += tsig2 * exp(-q1q2diff * q1q2diff / (2 * tls2))

    return kern
