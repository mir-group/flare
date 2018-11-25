from scipy.special import sph_harm
from scipy.special import spherical_in
from scipy.special import erf
from scipy.special import gamma
from scipy.integrate import quad
from scipy.linalg import cholesky
from numpy.linalg import inv
# import matplotlib.pyplot as plt
import numpy as np
import math


# -------------------------------------------------------------------
# basis functions: gaussians of variable sigma centered at the origin
# -------------------------------------------------------------------

def get_betas(n_tot, sigma_min, sigma_max):
    step = (sigma_max - sigma_min)/(n_tot - 1)
    betas = np.zeros([n_tot])

    for n in range(n_tot):
        sigma = sigma_min + step * n
        beta = 1 / (2 * sigma**2)
        betas[n] = beta
    return betas


def get_basis(rs, beta, l):
    basis_vals = rs**l * np.exp(-beta * rs**2)
    return basis_vals


def gauss_overlap(beta, beta_prime, l):
    overlap = (1/2) * (beta + beta_prime)**(-(3/2)-l) * gamma((3/2)+l)
    return overlap


def gauss_overlap_matrix(l, betas):
    overlap_matrix = np.zeros([len(betas), len(betas)])
    for m in range(len(betas)):
        beta = betas[m]
        for n in range(m, len(betas)):
            beta_prime = betas[n]
            overlap = gauss_overlap(beta, beta_prime, l)
            overlap_matrix[m, n] = overlap
            overlap_matrix[n, m] = overlap
    chol_mat = cholesky(overlap_matrix, lower=True)
    basis_mat = inv(chol_mat)
    return basis_mat


def return_ortho_val(r, n, l, betas, basis_mat):
    ortho_val = 0
    for m in range(len(betas)):
        beta = betas[m]
        ortho_val += basis_mat[n, m] * \
            get_basis(r, beta, l)
    return ortho_val


# -------------------------------------------------------------------
#                   compute radial integrals
# -------------------------------------------------------------------

def get_orth_ints(ri, l, alpha, betas, basis_mat):
    non_orth_ints = np.sqrt(np.pi / (16 * ri * alpha)) * \
        (((ri * alpha)**(l + (1/2))) / ((alpha + betas)**(l + (3/2)))) *\
        np.exp(-(alpha * betas * ri**2)/(alpha + betas))

    orth_ints = np.matmul(basis_mat, non_orth_ints)

    return orth_ints


def exact_radial(r, ri, alpha):
    return np.exp(-alpha * (r**2 + ri**2)) * spherical_in(l, 2*alpha*r*ri)


def reconstruct_radial(r, ri, l, betas, basis_mat, orth_ints):
    ortho_vals = np.zeros([len(betas)])
    for n in range(len(betas)):
        ortho_vals[n] = return_ortho_val(r, n, l, betas, basis_mat)

    radial_est = np.sum(orth_ints * ortho_vals)

    return radial_est

# -------------------------------------------------------------------
#                some preliminary formula checks
# -------------------------------------------------------------------


# Eq. (32) Bartok (2013)
# given a single atom i, return c_{lmi}(r)
def sph_coeff(atom_pos, r, l, m, alpha):
    ri = np.linalg.norm(atom_pos)
    theta = math.acos(atom_pos[2] / ri)
    phi = math.atan2(atom_pos[1], atom_pos[0])
    c_lm = 4 * np.pi * np.exp(-alpha * (r**2 + ri**2)) * \
        spherical_in(l, 2*alpha*r*ri) * \
        np.conj(sph_harm(m, l, phi, theta))

    return c_lm


# given a list of atoms, return c_{lmi}(r)
def radial_function(r, atom_pos_list, l_val, m_val, alpha):
    rad_val = 0
    for p in range(len(atom_pos_list)):
        atom_pos = atom_pos_list[p]
        rad_val += sph_coeff(atom_pos, r, l_val, m_val, alpha)
    return rad_val


def estimate_rho(pos, atom_pos_list, alpha, max_l):
    r = np.linalg.norm(pos)
    theta = math.acos(pos[2] / r)
    phi = math.atan2(pos[1], pos[0])

    rho_est = 0
    for p in range(len(atom_pos_list)):
        atom_pos = atom_pos_list[p]
        for l in range(max_l + 1):
            for m in range(-l, l+1):
                rho_est += sph_coeff(atom_pos, r, l, m, alpha) * \
                    sph_harm(m, l, phi, theta)

    return np.real(rho_est)


def rho_exact(pos, atom_pos_list, alpha):
    rho_ex = 0
    for n in range(len(atom_pos_list)):
        atom_pos = atom_pos_list[n]
        rho_ex += np.exp(-alpha * np.linalg.norm(pos - atom_pos)**2)
    return rho_ex


def basis_overlap(n, nprime, nmax, rcut, sig):
    # integral computed in Mathematica
    # full simplified version recorded here
    prefactor = (1 / (8 * nmax**2)) * sig * \
        np.exp(-((n**2 + nprime**2)*rcut**2) / (2 * nmax**2 * sig**2))

    term1 = 2 * nmax * rcut * sig * \
        (n + nprime - np.exp(((n - nmax + nprime) * rcut**2) /
         (nmax * sig**2)) * (n + 2 * nmax + nprime))

    term2 = np.exp(((n + nprime)**2*rcut**2) / (4 * nmax**2 * sig**2)) * \
        np.sqrt(np.pi) * ((n + nprime)**2 * rcut**2 + 2 * nmax**2 * sig**2)

    term3 = erf(((n + nprime) * rcut) / (2 * nmax * sig)) - \
        erf(((n - 2 * nmax + nprime) * rcut) / (2 * nmax * sig))

    overlap = prefactor * (term1 + term2 * term3)

    return overlap


def overlap_matrix(nmax, rcut, sig):
    overlap = np.zeros([nmax+1, nmax+1])
    for m in range(nmax + 1):
        for n in range(nmax + 1):
            overlap[m, n] = basis_overlap(m, n, nmax, rcut, sig)
    return overlap

if __name__ == '__main__':
    # test estimate_rho against rho_exact
    pos = np.array([2, 1.25, 2])
    atom_pos_list = [np.array([1, 2, 3]),
                     np.array([2, 2, 2])]
    alpha = 0.4
    max_l = 14

    rho_est = estimate_rho(pos, atom_pos_list, alpha, max_l)
    rho_ex = rho_exact(pos, atom_pos_list, alpha)

    assert(np.isclose(rho_est, rho_ex))

    # test get_betas
    n_tot = 5
    sigma_min = 0.1
    sigma_max = 5
    betas = get_betas(n_tot, sigma_min, sigma_max)
    assert(len(betas) == n_tot)
    assert(betas[0] == 1 / (2 * sigma_min**2))
    assert(betas[n_tot-1] == 1 / (2 * sigma_max**2))

    # test gauss_overlap_matrix and return_ortho_val
    # basis functions should be orthonormal
    n_tot = 5
    sigma_min = 0.1
    sigma_max = 5
    l = 0
    n = 4
    nprime = 3
    betas = get_betas(n_tot, sigma_min, sigma_max)
    basis_mat = gauss_overlap_matrix(l, betas)

    zer_test = quad(lambda x: x**2 *
                    return_ortho_val(x, n, l, betas, basis_mat) *
                    return_ortho_val(x, nprime, l, betas, basis_mat),
                    0, 30)[0]

    one_test = quad(lambda x: x**2 *
                    return_ortho_val(x, n, l, betas, basis_mat) *
                    return_ortho_val(x, n, l, betas, basis_mat),
                    0, 30)[0]

    assert(np.isclose(zer_test, 0))
    assert(np.isclose(one_test, 1))

    # test get_orth_ints
    # analytical and numerical calculations should agree
    ri = 2
    l = 0
    alpha = 0.4

    orth_ints = get_orth_ints(ri, l, alpha, betas, basis_mat)

    n = 2
    int_test = quad(lambda x: x**2 *
                    return_ortho_val(x, n, l, betas, basis_mat) *
                    exact_radial(x, ri, alpha),
                    0, 30)[0]
    assert(np.isclose(orth_ints[n], int_test))

    # test reconstruct_radial
    n_tot = 20
    sigma_min = 0.05
    sigma_max = 4
    l = 0
    ri = 4
    alpha = 0.4
    r = 4

    betas = get_betas(n_tot, sigma_min, sigma_max)
    basis_mat = gauss_overlap_matrix(l, betas)
    orth_ints = get_orth_ints(ri, l, alpha, betas, basis_mat)

    exact_val = exact_radial(r, ri, alpha)
    est_val = reconstruct_radial(r, ri, l, betas, basis_mat, orth_ints)
    assert(np.isclose(exact_val, est_val, 1e-4))
