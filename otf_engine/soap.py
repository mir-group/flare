from scipy.special import sph_harm
from scipy.special import spherical_in
from scipy.special import erf
from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import math


# -------------------------------------------------------------------
# basis functions: gaussians of variable sigma centered at the origin
# -------------------------------------------------------------------

def get_basis(rs, l, n, n_tot, sigma_min, sigma_max):
    sigma = sigma_min + ((sigma_max - sigma_min)/(n_tot - 1)) * n
    beta = 1 / (2 * sigma**2)
    basis_vals = rs**l * np.exp((-beta * rs**2))
    return basis_vals


def plot_basis(rs, l, n, n_tot, sigma_min, sigma_max):
    basis_vals = get_basis(rs, l, n, n_tot, sigma_min, sigma_max)
    plt.plot(rs, basis_vals)


def gauss_overlap(n, nprime, l, n_tot, sigma_min, sigma_max):
    step = (sigma_max - sigma_min)/(n_tot - 1)
    sigma = sigma_min + step * n
    sigma_prime = sigma_min + step * nprime

    beta = 1 / (2 * sigma**2)
    beta_prime = 1 / (2 * sigma_prime**2)

    overlap = (1/2) * (beta + beta_prime)**(-(3/2)-l) * gamma((3/2)+l)

    return overlap


def gauss_overlap_matrix(l, n_tot, sigma_min, sigma_max):
    overlap = np.zeros([n_tot, n_tot])
    for m in range(n_tot):
        for n in range(n_tot):
            overlap[m, n] = gauss_overlap(m, n, l, n_tot, sigma_min, sigma_max)
    return overlap


def plot_gauss_basis(rs, l, n, n_tot, sigma_min, sigma_max):
    comb_mat = gauss_overlap_matrix(l, n_tot, sigma_min, sigma_max)
    ortho_vals = np.zeros(len(rs))
    for m in range(n_tot):
        ortho_vals += comb_mat[n, m] * get_basis(rs, l, m, n_tot, sigma_min,
                                                 sigma_max)
    plt.plot(rs, ortho_vals)


def return_ortho_val(r, l, n, n_tot, sigma_min, sigma_max):
    comb_mat = gauss_overlap_matrix(l, n_tot, sigma_min, sigma_max)
    ortho_val = 0
    for m in range(n_tot):
        ortho_val += comb_mat[n, m] * get_basis(r, l, m, n_tot, sigma_min, 
                                                sigma_max)
    return ortho_val
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
    print(rho_est)

    rho_ex = rho_exact(pos, atom_pos_list, alpha)
    print(rho_ex)

    # test basis_overlap
    test_overlap = basis_overlap(4, 6, 14, 5, 0.5)
    print(test_overlap)
