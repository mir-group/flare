import math
from math import exp
import sys
import numpy as np
from scipy.linalg import solve_triangular
import time
sys.path.append('otf/otf_engine')
import env, gp, struc


def two_body_grad(env1, env2, d1, d2, hyps):
    return two_body_grad_from_env(env1.bond_array,
                                  env1.bond_types,
                                  env2.bond_array,
                                  env2.bond_types,
                                  d1, d2, hyps)


def two_body_grad_from_env(bond_array_1, bond_types_1, bond_array_2,
                           bond_types_2, d1, d2, hyps):
    sig = hyps[0]
    ls = hyps[1]
    S = sig * sig
    L = 1 / (ls * ls)
    sig_conv = 2 * sig
    ls_conv = -2 / (ls * ls * ls)

    kern = 0
    sig_derv = 0
    ls_derv = 0

    x1_len = len(bond_types_1)
    x2_len = len(bond_types_2)

    for m in range(x1_len):
        r1 = bond_array_1[m, 0]
        coord1 = bond_array_1[m, d1]
        typ1 = bond_types_1[m]

        for n in range(x2_len):
            r2 = bond_array_2[n, 0]
            coord2 = bond_array_2[n, d2]
            typ2 = bond_types_2[n]

            # check that bonds match
            if typ1 == typ2:
                rr = (r1-r2)*(r1-r2)
                kern += S*L*exp(-0.5*L*rr)*coord1*coord2*(1-L*rr)
                sig_derv += L*exp(-0.5*L*rr)*coord1*coord2*(1-L*rr) * sig_conv
                ls_derv += 0.5*coord1*coord2*S*exp(-L*rr/2) * \
                    (2+L*rr*(-5+L*rr))*ls_conv

    kern_grad = np.array([sig_derv, ls_derv])

    return kern, kern_grad


def get_likelihood_and_gradients(training_data, training_labels_np,
                                 kernel_grad, hyps, sigma_n):

    number_of_hyps = len(hyps)

    # initialize matrices
    size = len(training_data)*3
    k_mat = np.zeros([size, size])

    # add a matrix to include noise variance:
    hyp_mat = np.zeros([size, size, number_of_hyps+1])

    ds = [1, 2, 3]

    # calculate elements
    for m_index in range(size):
        x_1 = training_data[int(math.floor(m_index / 3))]
        d_1 = ds[m_index % 3]

        for n_index in range(m_index, size):
            x_2 = training_data[int(math.floor(n_index / 3))]
            d_2 = ds[n_index % 3]

            # calculate kernel and gradient
            cov = kernel_grad(x_1, x_2, d_1, d_2, hyps)

            # store kernel value
            k_mat[m_index, n_index] = cov[0]
            k_mat[n_index, m_index] = cov[0]

            # store gradients (excluding noise variance)
            for p_index in range(number_of_hyps):
                hyp_mat[m_index, n_index, p_index] = cov[1][p_index]
                hyp_mat[n_index, m_index, p_index] = cov[1][p_index]

    # add gradient of noise variance
    hyp_mat[:, :, number_of_hyps] = np.eye(size) * 2 * sigma_n

    # matrix manipulation
    ky_mat = k_mat + sigma_n ** 2 * np.eye(size)
    ky_mat_inv = np.linalg.inv(ky_mat)
    alpha = np.matmul(ky_mat_inv, training_labels_np)
    alpha_mat = np.matmul(alpha.reshape(alpha.shape[0], 1),
                          alpha.reshape(1, alpha.shape[0]))
    like_mat = alpha_mat - ky_mat_inv

    # calculate likelihood
    like = (-0.5*np.matmul(training_labels_np, alpha) -
            0.5*math.log(np.linalg.det(ky_mat)) -
            math.log(2 * np.pi) * k_mat.shape[1] / 2)

    # calculate likelihood gradient
    like_grad = np.zeros(number_of_hyps + 1)
    for n in range(number_of_hyps + 1):
        like_grad[n] = 0.5 * np.trace(np.matmul(like_mat, hyp_mat[:, :, n]))

    return like, like_grad


def likelihood_ascent(training_data, training_labels_np, kernel_grad,
                      old_hyps, old_sigma_n, step_factor, max_steps, tol,
                      verbose=False):
    number_of_hyps = len(old_hyps) # excluding noise
    new_hyps = old_hyps
    new_sigma_n = old_sigma_n
    old_like = 1e16

    for _ in range(max_steps):
        like, like_grad = get_likelihood_and_gradients(training_data,
                                                       training_labels_np,
                                                       kernel_grad, new_hyps,
                                                       new_sigma_n)
        new_hyps = new_hyps + like_grad[0:number_of_hyps] * step_factor
        new_sigma_n = new_sigma_n + like_grad[number_of_hyps] * step_factor

        # check tolerance
        if (like > old_like) and (like - old_like < tol):
            print(like-old_like)
            break

        old_like = like

        if verbose:
            print(like)
            print(_)

    return new_hyps, new_sigma_n, like


# testing ground
if __name__ == '__main__':
    # make env1
    cell = np.eye(3)
    cutoff = np.linalg.norm(np.array([0.5, 0.5, 0.5])) + 0.001

    positions_1 = [np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])]
    species_1 = ['B', 'A']
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1, cutoff)
    env1 = env.ChemicalEnvironment(test_structure_1, atom_1)

    # make env 2
    positions_2 = [np.array([0, 0, 0]), np.array([0.25, 0.3, 0.4])]
    species_2 = ['B', 'A']
    atom_2 = 0
    test_structure_2 = struc.Structure(cell, species_2, positions_2, cutoff)
    env2 = env.ChemicalEnvironment(test_structure_2, atom_2)

    # set hyperparameters
    d1 = 1
    d2 = 1
    sig = 3.1415
    ls = 1
    hyp = np.array([sig, ls])

    # check if kernel matches old kernel
    assert(np.isclose(env.two_body(env1, env2, d1, d2, sig, ls),
           two_body_grad(env1, env2, d1, d2, hyp)[0]))

    # check if two body derivative formulas are correct
    kern, kern_grad = two_body_grad(env1, env2, d1, d2, hyp)
    S_derv = kern_grad[0]
    L_derv = kern_grad[1]

    delta = 1e-8
    tol = 1e-5
    new_sig = sig + delta
    new_ls = ls + delta

    S_derv_brute = (env.two_body(env1, env2, d1, d2, new_sig, ls) -
                    env.two_body(env1, env2, d1, d2, sig, ls)) / delta

    L_derv_brute = (env.two_body(env1, env2, d1, d2, sig, new_ls) -
                    env.two_body(env1, env2, d1, d2, sig, ls)) / delta

    assert(np.isclose(S_derv, S_derv_brute, tol))
    assert(np.isclose(L_derv, L_derv_brute, tol))

    # check that new likelihood matches old likelihood
    sig_n = 5
    forces = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    gp_test = gp.GaussianProcess('two_body')
    hyp_gp = [sig, ls, sig_n]
    gp_test.update_db(test_structure_1, forces)
    gp_like = gp_test.like_hyp(hyp_gp)

    tb = gp_test.training_data
    training_labels_np = gp_test.training_labels_np
    like, like_grad = get_likelihood_and_gradients(tb, training_labels_np,
                                                   two_body_grad, hyp,
                                                   sig_n)
    new_like = like

    assert(gp_like == new_like)

    # check that likelihood gradient is correct
    delta = 1e-7
    new_sig = np.array([sig + delta, ls])
    new_ls = np.array([sig, ls + delta])
    new_n = sig_n + delta

    sig_grad_brute = (get_likelihood_and_gradients(tb, training_labels_np,
                                                   two_body_grad, new_sig,
                                                   sig_n)[0] -
                      get_likelihood_and_gradients(tb, training_labels_np,
                                                   two_body_grad, hyp,
                                                   sig_n)[0])\
        / delta

    ls_grad_brute = (get_likelihood_and_gradients(tb, training_labels_np,
                                                  two_body_grad, new_ls,
                                                  sig_n)[0] -
                     get_likelihood_and_gradients(tb, training_labels_np,
                                                  two_body_grad, hyp,
                                                  sig_n)[0])\
        / delta

    n_grad_brute = (get_likelihood_and_gradients(tb, training_labels_np,
                                                 two_body_grad, hyp,
                                                 new_n)[0] -
                    get_likelihood_and_gradients(tb, training_labels_np,
                                                 two_body_grad, hyp,
                                                 sig_n)[0])\
        / delta

    tol = 1e-3
    assert(np.isclose(like_grad[0], sig_grad_brute, tol))
    assert(np.isclose(like_grad[1], ls_grad_brute))
    assert(np.isclose(like_grad[2], n_grad_brute))

    # test likelihood ascent
    hyp = np.array([10, 10])
    sig_n = 3.89
    step_factor = 0.1
    max_steps = 1000
    tol = 1e-8
    new_hyps, new_sigma_n, like = likelihood_ascent(tb, training_labels_np, 
                                                    two_body_grad,
                                                    hyp, sig_n, step_factor, 
                                                    max_steps, tol, True)

    gp_test.train()
    print(new_hyps)
    print(new_sigma_n)
    print(like)
    print(gp_test.sigma_f)
    print(gp_test.length_scale)
    print(gp_test.sigma_n)
    print(gp_test.get_likelihood())
