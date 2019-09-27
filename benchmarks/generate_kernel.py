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

    num_atoms = 100
    num_envs = 10

    # Lets make the file
    filename = os.path.join(args.location, "kernel_data.hdf5")
    file = h5py.File(filename, 'w')


    # Let's construct each environment
    cutoffs = np.array([1, 1])
    cell = np.eye(3)
    species = [1, 2, 1]
    atom = 0
    for i in range(num_envs):
        positions = np.random.random((num_atoms, 3))
        positions[0, :] = np.zeros(3)
        test_structure = struc.Structure(cell, species, positions)
        environment = env.AtomicEnvironment(test_structure, atom, cutoffs)

        add_environment(file, f"env{i}", environment)  # Add them to the file

    # set hyperparameters
    sig1 = np.random.random()
    ls1 = np.random.random()

    add_parameters(file, sig1, ls1, cutoffs[0])
    add_metadata(file, num_envs)

    file.close()


def add_environment(file, name, environment):
    h5_env1 = file.create_group(name)
    h5_env1.create_dataset("bond_array_2", data=environment.bond_array_2)
    h5_env1.create_dataset("bond_array_2_T", data=environment.bond_array_2.T)


def add_parameters(file, sig, ls, r_cut):
    parameters = file.create_group("parameters")
    parameters.create_dataset("sig", data=sig)
    parameters.create_dataset("ls", data=ls)
    parameters.create_dataset("r_cut", data=r_cut)

def add_metadata(file, num_envs):
    parameters = file.create_group("metadata")
    parameters.create_dataset("num_envs", data=num_envs)
