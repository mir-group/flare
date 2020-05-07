"""
Helper functions which obtain forces and energies
corresponding to atoms in structures. These functions automatically
cast atoms into their respective atomic environments.
"""
import numpy as np
import multiprocessing as mp

from typing import Tuple
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.struc import Structure


def predict_on_atom(param: Tuple[Structure, int, GaussianProcess]) -> (
        'np.ndarray', 'np.ndarray'):
    """
    Return the forces/std. dev. uncertainty associated with an individual atom
    in a structure, without necessarily having cast it to a chemical
    environment. In order to work with other functions,
    all arguments are passed in as a tuple.

    :param param: tuple of FLARE Structure, atom index, and Gaussian Process
        object
    :type param: Tuple(Structure, integer, GaussianProcess)
    :return: 3-element force array and associated uncertainties
    :rtype: (np.ndarray, np.ndarray)
    """
    # Unpack the input tuple, convert a chemical environment
    structure, atom, gp = param
    # Obtain the associated chemical environment
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs)
    components = []
    stds = []
    # predict force components and standard deviations
    for i in range(3):
        force, var = gp.predict(chemenv, i + 1)
        components.append(float(force))
        stds.append(np.sqrt(np.abs(var)))

    return np.array(components), np.array(stds)


def predict_on_atom_en(param: Tuple[Structure, int, GaussianProcess]) -> (
        'np.ndarray', 'np.ndarray', float):
    """
    Return the forces/std. dev. uncertainty / energy associated with an
    individual atom in a structure, without necessarily having cast it to a
    chemical environment. In order to work with other functions,
    all arguments are passed in as a tuple.

    :param param: tuple of FLARE Structure, atom index, and Gaussian Process
        object
    :type param: Tuple(Structure, integer, GaussianProcess)
    :return: 3-element force array, associated uncertainties, and local energy
    :rtype: (np.ndarray, np.ndarray, float)
    """
    # Unpack the input tuple, convert a chemical environment
    structure, atom, gp = param
    # Obtain the associated chemical environment
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs)
    comps = []
    stds = []
    # predict force components and standard deviations
    for i in range(3):
        force, var = gp.predict(chemenv, i + 1)
        comps.append(float(force))
        stds.append(np.sqrt(np.abs(var)))

    # predict local energy
    local_energy = gp.predict_local_energy(chemenv)

    return np.array(comps), np.array(stds), local_energy


def predict_on_atom_en_std(param):
    """Predict local energy and predictive std of a chemical environment."""

    structure, atom, gp = param
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs)

    # predict local energy
    loc_en, loc_en_var = gp.predict_local_energy_and_var(chemenv)
    loc_en_std = np.sqrt(np.abs(loc_en_var))

    return loc_en, loc_en_std


def predict_on_structure(structure: Structure, gp: GaussianProcess,
                         n_cpus: int=None, write_to_structure: bool = True) \
        -> ('np.ndarray', 'np.ndarray'):
    """
    Return the forces/std. dev. uncertainty associated with each
    individual atom in a structure. Forces are stored directly to the
    structure and are also returned.

    :param structure: FLARE structure to obtain forces for, with N atoms
    :param gp: Gaussian Process model
    :param write_to_structure: Write results to structure's forces,
                            std attributes
    :return: N x 3 numpy array of foces, Nx3 numpy array of uncertainties
    :rtype: (np.ndarray, np.ndarray)
    """

    forces = np.zeros((structure.nat, 3))
    stds = np.zeros((structure.nat, 3))

    forces = np.zeros(shape=(structure.nat,3))
    stds = np.zeros(shape=(structure.nat,3))

    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            forces[n][i] = float(force)
            stds[n][i] = float(np.sqrt(np.absolute(var)))

            if write_to_structure:
                structure.forces[n][i] = float(force)
                structure.stds[n][i] = np.sqrt(np.abs(var))

    return forces, stds


def predict_on_structure_par(structure: Structure,
                             gp: GaussianProcess,
                             n_cpus: int = None,
                             write_to_structure: bool = True) -> (
        'np.ndarray', 'np.ndarray'):
    """
    Return the forces/std. dev. uncertainty associated with each
    individual atom in a structure. Forces are stored directly to the
    structure and are also returned.

    :param structure: FLARE structure to obtain forces for, with N atoms
    :param gp: Gaussian Process model
    :param n_cpus: Number of cores to parallelize over
    :param write_to_structure: Write results to structure's forces,
                            std attributes
    :return: N x 3 array of forces, N x 3 array of uncertainties
    :rtype: (np.ndarray, np.ndarray)
    """
    # Just work in serial in the number of cpus is 1
    if n_cpus is 1:
        return predict_on_structure(structure, gp,n_cpus,write_to_structure)

    # Automatically detect number of cpus available
    if (n_cpus is None):
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=n_cpus)

    forces = np.zeros((structure.nat, 3))
    stds = np.zeros((structure.nat, 3))

    # Parallelize over atoms in structure
    results = []
    for atom in range(structure.nat):
        results.append(pool.apply_async(predict_on_atom,
                                        args=[(structure, atom, gp)]))
    pool.close()
    pool.join()
    forces = np.zeros(shape=(structure.nat,3))
    stds = np.zeros(shape=(structure.nat,3))

    for i in range(structure.nat):
        r = results[i].get()
        forces[i] = r[0]
        stds[i] = r[1]
        if write_to_structure:
            structure.forces[i] = r[0]
            structure.stds[i] = r[1]

    return forces, stds


def predict_on_structure_en(structure: Structure, gp: GaussianProcess,
                            n_cpus: int = None) -> (
        'np.ndarray', 'np.ndarray', 'np.ndarray'):
    """
    Return the forces/std. dev. uncertainty / local energy associated with each
    individual atom in a structure. Forces are stored directly to the
    structure and are also returned.

    :param structure: FLARE structure to obtain forces for, with N atoms
    :param gp: Gaussian Process model
    :param n_cpus: Dummy parameter passed as an argument to allow for
        flexibility when the callable may or may not be parallelized
    :return: N x 3 array of forces, N x 3 array of uncertainties,
        N-length array of energies
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    # Set up local energy array
    forces = np.zeros((structure.nat, 3))
    stds = np.zeros((structure.nat, 3))
    local_energies = np.zeros(structure.nat)

    # Loop through atoms in structure and predict forces, uncertainties,
    # and energies
    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            forces[n][i] = float(force)
            stds[n][i] = np.sqrt(np.abs(var))
        local_energies[n] = gp.predict_local_energy(chemenv)

    structure.forces = forces
    structure.stds = stds

    return forces, stds, local_energies


def predict_on_structure_par_en(structure: Structure, gp: GaussianProcess,
                                n_cpus: int = None) -> (
        'np.ndarray', 'np.ndarray', 'np.ndarray'):
    """
    Return the forces/std. dev. uncertainty / local energy associated with each
    individual atom in a structure, parallelized over atoms. Forces are
    stored directly to the structure and are also returned.

    :param structure: FLARE structure to obtain forces for, with N atoms
    :param gp: Gaussian Process model
    :param n_cpus: Number of cores to parallelize over
    :return: N x 3 array of forces, N x 3 array of uncertainties,
        N-length array of energies
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    # Work in serial if the number of cpus is 1
    if n_cpus is 1:
        predict_on_structure_en(structure, gp)

    forces = np.zeros((structure.nat, 3))
    stds = np.zeros((structure.nat, 3))
    local_energies = np.zeros(structure.nat)

    if n_cpus is None:
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=n_cpus)

    results = []
    # Parallelize over atoms in structure
    for atom_i in range(structure.nat):
        results.append(pool.apply_async(predict_on_atom_en,
                                        args=[(structure, atom_i, gp)]))
    pool.close()
    pool.join()

    # Compile results
    for i in range(structure.nat):
        r = results[i].get()
        forces[i] = r[0]
        stds[i] = r[1]
        local_energies[i] = r[2]

    structure.forces = forces
    structure.stds = stds

    return forces, stds, local_energies
