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
    sig, ls, r_cut = read_parameters(file)
    num_envs = read_metadata(file)

    bond_arrays = []
    for i in range(num_envs):
        bond_arrays.append(read_environment(file, f"env{i}", "bond_array_2"))

    file.close()

    start = time.time()
    en.two_body_force_en_jit(bond_arrays[0], bond_arrays[1], 1, sig, ls, r_cut,
                             cutoff_func)
    end = time.time()
    print("Compile and first two body runtime", end - start)

    kern = 0
    start = time.time()
    for i in range(num_envs):
        for j in range(i+1, num_envs):
            for dim in range(1, 4):
                kern += en.two_body_force_en_jit(bond_arrays[i],
                                                 bond_arrays[j], dim, sig, ls,
                                                 r_cut, cutoff_func)

    end = time.time()
    print("Total two body runtime", end - start)
    print("Total kernel value", kern)


def read_environment(file, name, dataset):
    var = np.zeros(file[name][dataset].shape, dtype=np.float32)
    file[name][dataset].read_direct(var)

    return var


def read_parameters(file):
    sig = file["parameters"]['sig'][()]
    ls = file["parameters"]['ls'][()]
    r_cut = file["parameters"]['r_cut'][()]

    return sig, ls, r_cut


def read_metadata(file):
    num_envs = file['metadata']['num_envs'][()]

    return num_envs
