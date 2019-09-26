import os

import numpy as np
import h5py

from flare import env, gp, struc
import flare.kernels as en
import flare.cutoffs as cf


def generate_kernel(args):
    if args.subtype == "two-body":
        generate_two_body(args)


def generate_two_body(args):
    """Check that the analytical force/en kernel matches finite difference of
    energy kernel."""

    num_atoms = 1000

    # create env 1
    cell = np.eye(3)
    cutoffs = np.array([1, 1])

    positions_1 = np.random.random((num_atoms, 3))
    positions_1[0, :] = np.zeros(3)

    species_1 = [1, 2, 1]
    atom_1 = 0
    test_structure_1 = struc.Structure(cell, species_1, positions_1)

    env1 = env.AtomicEnvironment(test_structure_1, atom_1, cutoffs)

    # create env 2
    positions_2 = np.random.random((num_atoms, 3))
    positions_2[0, :] = np.zeros(3)

    species_2 = [1, 1, 2]
    atom_2 = 0
    test_structure_1 = struc.Structure(cell, species_2, positions_2)
    env2 = env.AtomicEnvironment(test_structure_1, atom_2, cutoffs)

    # set hyperparameters
    sig1 = np.random.random()
    ls1 = np.random.random()
    d1 = 1

    filename = os.path.join(args.location, "kernel_data.hdf5")
    file = h5py.File(filename, 'w')

    add_environment(file, "env1", env1)
    add_environment(file, "env2", env2)
    add_parameters(file, d1, sig1, ls1, cutoffs[0])

    file.close()


def add_environment(file, name, environment):
    h5_env1 = file.create_group(name)
    h5_env1.create_dataset("bond_array_2", data=environment.bond_array_2)
    h5_env1.create_dataset("bond_array_2_T", data=environment.bond_array_2.T)


def add_parameters(file, d1, sig, ls, r_cut):
    parameters = file.create_group("parameters")
    parameters.create_dataset("d1", data=d1)
    parameters.create_dataset("sig", data=sig)
    parameters.create_dataset("ls", data=ls)
    parameters.create_dataset("r_cut", data=r_cut)
