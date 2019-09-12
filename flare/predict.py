"""
Helper functions which obtain forces and energies
corresponding to atoms in structures
"""
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.struc import Structure

import numpy as np
import concurrent.futures


def predict_on_atom(structure: Structure, atom: int, gp: GaussianProcess):
    """
    Return the forces/std. dev. uncertainty associated with an atom in a
    structure
    :param structure:
    :param atom:
    :param gp:
    :return:
    """
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs)
    components = []
    stds = []
    # predict force components and standard deviations
    for i in range(3):
        force, var = gp.predict(chemenv, i + 1)
        components.append(float(force))
        stds.append(np.sqrt(np.abs(var)))

    return np.array(components), np.array(stds)


def predict_on_atom_en(structure: Structure, atom: int, gp: GaussianProcess):
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


def predict_on_structure_par(structure: Structure, gp: GaussianProcess):
    n = 0
    atom_list = [(structure, atom, gp) for atom in range(structure.nat)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(predict_on_atom, atom_list):
            for i in range(3):
                structure.forces[n][i] = res[0][i]
                structure.stds[n][i] = res[1][i]
            n += 1
    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds


def predict_on_structure_par_en(structure: Structure, gp: GaussianProcess):
    n = 0
    atom_list = [(structure, atom, gp) for atom in range(structure.nat)]
    local_energies = [0 for n in range(structure.nat)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(predict_on_atom_en, atom_list):
            for i in range(3):
                structure.forces[n][i] = res[0][i]
                structure.stds[n][i] = res[1][i]
            local_energies[n] = res[2]
            n += 1
    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds, local_energies


def predict_on_structure(structure: Structure, gp: GaussianProcess):
    for n in range(structure.nat):
        chemenv = AtomicEnvironment(structure, n, gp.cutoffs)
        for i in range(3):
            force, var = gp.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.abs(var))
    forces = np.array(structure.forces)
    stds = np.array(structure.stds)
    return forces, stds


def predict_on_structure_en(structure: Structure, gp: GaussianProcess):
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