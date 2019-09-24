import time
import os

import numpy as np
import h5py

from flare import env, gp, struc
import flare.kernels as en
import flare.cutoffs as cf


def run_kernel(args):
    """
    Runs a kernel benchmark using the args.subtype argument to choose the
    given benchmark
    """
    if args.subtype == "two-body":
        run_two_body(args)


def run_two_body(args):
    cutoff_func = cf.quadratic_cutoff

    filename = os.path.join(args.location, "kernel_data.hdf5")
    file = h5py.File(filename, 'r')

    bond_array_1 = read_environment(file, "env1", "bond_array_2")
    bond_array_2 = read_environment(file, "env2", "bond_array_2")
    d1, sig, ls, r_cut = read_parameters(file)

    file.close()

    start = time.time()
    en.two_body_force_en_jit(bond_array_1, bond_array_2, d1, sig, ls, r_cut,
                             cutoff_func)
    end = time.time()
    print("First two body runtime", end - start)

    start = time.time()
    en.two_body_force_en_jit(bond_array_1, bond_array_2, d1, sig, ls, r_cut,
                             cutoff_func)
    end = time.time()
    print("Second two body runtime", end - start)


def read_environment(file, name, dataset):
    var = np.zeros(file[name][dataset].shape, dtype=np.float32)
    file[name][dataset].read_direct(var)

    return var


def read_parameters(file):
    d1 = file["parameters"]['d1'][()]
    sig = file["parameters"]['sig'][()]
    ls = file["parameters"]['ls'][()]
    r_cut = file["parameters"]['r_cut'][()]

    return d1, sig, ls, r_cut
