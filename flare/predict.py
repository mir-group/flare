"""
Helper functions which obtain forces and energies
corresponding to atoms in structures. These functions automatically
cast atoms into their respective atomic environments.
"""
import numpy as np
import multiprocessing as mp

from typing import Tuple, List
from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.struc import Structure


def predict_on_atom(param: Tuple[Structure, int, GaussianProcess], energy=False) -> (
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
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs,
                                cutoffs_mask=gp.hyps_mask)
    # predict force components and standard deviations
    force, var = gp.predict(chemenv)

    if energy:
        # predict local energy
        local_energy = gp.predict_local_energy(chemenv)
        return force, np.sqrt(np.abs(var)), local_energy

    return force, np.sqrt(np.abs(var))


def predict_on_atom_en_std(param):
    """Predict local energy and predictive std of a chemical environment."""

    structure, atom, gp = param
    chemenv = AtomicEnvironment(structure, atom, gp.cutoffs,
                                cutoffs_mask=gp.hyps_mask)

    # predict local energy
    loc_en, loc_en_var = gp.predict_local_energy_and_var(chemenv)
    loc_en_std = np.sqrt(np.abs(loc_en_var))

    return loc_en, loc_en_std


def predict_on_structure(structure: Structure, gp: GaussianProcess,
                         n_cpus: int = None,
                         write_to_structure: bool = True,
                         selective_atoms: List[int] = None,
                         skipped_atom_value=0, energy=False) -> (
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
    forces = np.zeros(shape=(structure.nat, 3))
    stds = np.zeros(shape=(structure.nat, 3))

    if selective_atoms:
        forces.fill(skipped_atom_value)
        stds.fill(skipped_atom_value)
        local_energies.fill(skipped_atom_value)
    else:
        selective_atoms = []

    if write_to_structure and energy:
        if structure.local_energies is None:
            structure.local_energies = np.zeros(structure.nat)

    # Loop through atoms in structure and predict forces, uncertainties,
    # and energies
    for n in range(structure.nat):

        if selective_atoms and n not in selective_atoms:
            continue

        chemenv = AtomicEnvironment(structure, n, gp.cutoffs,
                                    cutoffs_mask=gp.hyps_mask)

        force, var = gp.predict(chemenv)
        forces[n] = force
        stds[n] = np.sqrt(np.abs(var))

        if write_to_structure and structure.forces is not None:
            structure.forces[n] = force
            structure.stds[n] = np.sqrt(np.abs(var))

        if energy:
            local_energies[n] = gp.predict_local_energy(chemenv)
            if write_to_structure:
                structure.local_energies[n] = local_energies[n]

    if energy:
        return forces, stds, local_energies

    return forces, stds


def predict_on_structure_par(structure: Structure, gp: GaussianProcess,
                             n_cpus: int = None,
                             write_to_structure: bool = True,
                             selective_atoms: List[int] = None,
                             skipped_atom_value=0, energy=False) -> (
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
    if n_cpus == 1:
        return predict_on_structure(structure, gp,
                                    n_cpus, write_to_structure,
                                    selective_atoms,
                                    skipped_atom_value,
                                    energy)

    forces = np.zeros((structure.nat, 3))
    stds = np.zeros((structure.nat, 3))
    local_energies = np.zeros(structure.nat)
    forces = np.zeros(shape=(structure.nat, 3))
    stds = np.zeros(shape=(structure.nat, 3))

    if selective_atoms:
        forces.fill(skipped_atom_value)
        stds.fill(skipped_atom_value)
        local_energies.fill(skipped_atom_value)
    else:
        selective_atoms = []

    if n_cpus is None:
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=n_cpus)

    # Parallelize over atoms in structure
    results = []
    for atom_i in range(structure.nat):

        if atom_i not in selective_atoms and selective_atoms:
            results.append(None)
            continue

        results.append(pool.apply_async(predict_on_atom,
                                        args=[(structure, atom_i, gp), True]))
    pool.close()
    pool.join()

    # Compile results
    for i in range(structure.nat):

        if i not in selective_atoms and selective_atoms:
            continue

        r = results[i].get()
        forces[i][:] = r[0]
        stds[i][:] = r[1]
        if energy:
            local_energies[i] = r[2]

        if write_to_structure:
            structure.forces[i] = forces[i]
            structure.stds[i] = stds[i]

    if energy:
        if write_to_structure:
            structure.local_energies = np.copy(local_energies)
        return forces, stds, local_energies
    return forces, stds



def predict_on_atom_mgp(atom: int, structure, mgp,
                        write_to_structure=False):
    chemenv = AtomicEnvironment(structure, atom, mgp.cutoffs,
                                cutoffs_mask=mgp.hyps_mask)
    # predict force components and standard deviations
    force, var, virial, local_energy = mgp.predict(chemenv)
    comps = force
    stds = np.sqrt(np.absolute(var))

    if write_to_structure:
        structure.forces[atom][:] = force
        structure.stds[atom][:] = stds
        structure.local_energies[atom] = local_energy

    return comps, stds, local_energy


def predict_on_structure_mgp(structure, mgp, output=None,
                             output_name=None, n_cpus=None,
                             write_to_structure=True,
                             selective_atoms: List[int] = None,
                             skipped_atom_value=0):  # changed
    """
    Assign forces to structure based on an mgp
    """
    if output and output_name:
        output.write_to_output('\npredict with mapping:\n', output_name)

    forces = np.zeros(shape=(structure.nat, 3))
    stds = np.zeros(shape=(structure.nat, 3))
    energy = np.zeros(shape=(structure.nat))
    if write_to_structure and structure.local_energies is None:
        structure.local_energies = np.zeros(shape=(structure.nat))

    if selective_atoms:
        forces.fill(skipped_atom_value)
        stds.fill(skipped_atom_value)
    else:
        selective_atoms = []

    for n in range(structure.nat):

        if n not in selective_atoms and selective_atoms:
            continue

        forces[n, :], stds[n, :], energy[n] = \
            predict_on_atom_mgp(n, structure, mgp,
                                write_to_structure)

    return forces, stds, energy
