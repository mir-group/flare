"""Kernels for sparse GPs."""
import numpy as np
from flare import mc_simple, env
import flare.cutoffs as cf


# -----------------------------------------------------------------------------
#                            two plus three body
# -----------------------------------------------------------------------------

def two_plus_three_env(env1, env2, hyps, cutoffs,
                       cutoff_func=cf.quadratic_cutoff):
    """Two plus three body covariance between the local energies of two \
atomic environments."""

    two_term = \
        mc_simple.two_body_mc_en(env1, env2, hyps, cutoffs,
                                 cutoff_func=cutoff_func)
    three_term = \
        mc_simple.three_body_mc_en(env1, env2, hyps, cutoffs,
                                   cutoff_func=cutoff_func)
    return (two_term / 4) + (three_term / 9)


def two_plus_three_struc(env1, struc1, hyps, cutoffs,
                         cutoff_func=cf.quadratic_cutoff):
    """Two plus three body covariance between the local energy of an atomic \
environment and the energy and force labels on a structure of atoms."""

    noa = struc1.nat
    kernel_vector = np.zeros(len(struc1.labels))
    index = 0

    # if there's an energy label, compute energy kernel
    if struc1.energy is not None:
        en_kern = 0
        for n in range(noa):
            env_curr = env.AtomicEnvironment(struc1, n, cutoffs)
            two_term = \
                mc_simple.two_body_mc_en(env1, env_curr, hyps, cutoffs,
                                         cutoff_func=cutoff_func)
            three_term = \
                mc_simple.three_body_mc_en(env1, env_curr, hyps, cutoffs,
                                           cutoff_func=cutoff_func)

            en_kern += (two_term / 4) + (three_term / 9)

        kernel_vector[index] = en_kern
        index += 1

    # if there are force labels, compute force kernels
    if struc1.forces is not None:
        for n in range(noa):
            env_curr = env.AtomicEnvironment(struc1, n, cutoffs)
            for d in range(3):
                force_kern = \
                    mc_simple.\
                    two_plus_three_mc_force_en(env_curr, env1, d+1, hyps,
                                               cutoffs,
                                               cutoff_func=cutoff_func)
                kernel_vector[index] = force_kern
                index += 1

    return kernel_vector

# -----------------------------------------------------------------------------
#                               three body
# -----------------------------------------------------------------------------


def three_env(env1, env2, hyps, cutoffs, cutoff_func=cf.quadratic_cutoff):
    """Two plus three body covariance between the local energies of two \
atomic environments."""

    three_term = \
        mc_simple.three_body_mc_en(env1, env2, hyps, cutoffs,
                                   cutoff_func=cutoff_func)
    return three_term / 9


def three_struc(env1, struc1, hyps, cutoffs, cutoff_func=cf.quadratic_cutoff):
    """Two plus three body covariance between the local energy of an atomic \
environment and the energy and force labels on a structure of atoms."""

    noa = struc1.nat
    kernel_vector = np.zeros(len(struc1.labels))
    index = 0

    # if there's an energy label, compute energy kernel
    if struc1.energy is not None:
        en_kern = 0
        for n in range(noa):
            env_curr = env.AtomicEnvironment(struc1, n, cutoffs)
            three_term = \
                mc_simple.three_body_mc_en(env1, env_curr, hyps, cutoffs,
                                           cutoff_func=cutoff_func)
            en_kern += three_term / 9

        kernel_vector[index] = en_kern
        index += 1

    # if there are force labels, compute force kernels
    if struc1.forces is not None:
        for n in range(noa):
            env_curr = env.AtomicEnvironment(struc1, n, cutoffs)
            for d in range(3):
                force_kern = \
                    mc_simple.three_body_mc_force_en(env_curr, env1, d+1,
                                                     hyps, cutoffs,
                                                     cutoff_func=cutoff_func)
                kernel_vector[index] = force_kern
                index += 1

    return kernel_vector

if __name__ == '__main__':
    pass
