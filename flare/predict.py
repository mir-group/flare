"""
Helper functions which obtain forces and energies
corresponding to atoms in structures
"""
import numpy as np
import multiprocessing as mp

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.struc import Structure

def predict_on_atom(param):
    """
    Return the forces/std. dev. uncertainty associated with an atom in a
    structure
    :param param: tuple of structure, atom, and gp
    :return:
    """
    structure, atom, gp = param
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs)
    components = []
    stds = []
    # predict force components and standard deviations
    for i in range(3):
        force, var = gp.predict(chemenv, i + 1)
        components.append(float(force))
        stds.append(np.sqrt(np.abs(var)))

    return np.array(components), np.array(stds)

def predict_on_atom_en(param):
    """
    Return ...
    :param param: tuple of structure, atom, and gp
    :return:
    """
    structure, atom, gp = param
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
    return comps, stds, local_energy


def predict_on_structure_par(structure: Structure, gp: GaussianProcess, no_cpus=None):

    if (no_cpus is 1):
        predict_on_structure(structure, gp)

    if (no_cpus is None):
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=no_cpus)
    results = []

    for atom in range(structure.nat):
        results.append(pool.apply_async(predict_on_atom,
            args=[(structure, atom, gp)]))
    pool.close()
    pool.join()

    for i in range(structure.nat):
        r = results[i].get()
        structure.forces[i] = r[0]
        structure.stds[i] = r[1]

    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds


def predict_on_structure_par_en(structure: Structure, gp: GaussianProcess, no_cpus=None):

    if (no_cpus is 1):
        predict_on_structure_en(structure, gp)

    atom_list = [(structure, atom, gp) for atom in range(structure.nat)]
    local_energies = [0 for n in range(structure.nat)]

    results = []
    if (no_cpus is None):
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=no_cpus)
    for atom in range(structure.nat):
        results.append(pool.apply_async(predict_on_atom_en,
            args=[(structure, atom, gp)]))
    pool.close()
    pool.join()

    for i in range(structure.nat):
        r = results[i].get()
        structure.forces[i] = r[0]
        structure.stds[i] = r[1]
        local_energies[i] = r[2]

    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds, local_energies


def predict_on_structure(structure: Structure, gp: GaussianProcess, no_cpus=None):
    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.abs(var))
    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds


def predict_on_structure_en(structure: Structure, gp: GaussianProcess, no_cpus=None):
    local_energies = [0 for _ in range(structure.nat)]

    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.abs(var))
        local_energies[n] = gp.predict_local_energy(chemenv)

    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds, local_energies
