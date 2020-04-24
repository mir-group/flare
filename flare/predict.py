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
from flare.gp_algebra import get_kernel_mat, get_en_kernel_mat
from mpi4py import MPI


def predict_on_atom(
    param: Tuple[Structure, int, GaussianProcess]
) -> ("np.ndarray", "np.ndarray"):
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


def predict_on_atom_en(
    param: Tuple[Structure, int, GaussianProcess]
) -> ("np.ndarray", "np.ndarray", float):
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


def predict_on_structure(
    structure: Structure,
    gp: GaussianProcess,
    n_cpus: int = None,
    write_to_structure: bool = True,
    selective_atoms: List[int] = None,
    skipped_atom_value=0,
) -> ("np.ndarray", "np.ndarray"):
    """
    Return the forces/std. dev. uncertainty associated with each
    individual atom in a structure. Forces are stored directly to the
    structure and are also returned.

    :param structure: FLARE structure to obtain forces for, with N atoms
    :param gp: Gaussian Process model
    :param write_to_structure: Write results to structure's forces,
                            std attributes
    :param selective_atoms: Only predict on these atoms; e.g. [0,1,2] will
                                only predict and return for those atoms
    :param skipped_atom_value: What value to use for atoms that are skipped.
            Defaults to 0 but other options could be e.g. NaN. Will NOT
            write this to the structure if write_to_structure is True.
    :return: N x 3 numpy array of foces, Nx3 numpy array of uncertainties
    :rtype: (np.ndarray, np.ndarray)
    """

    N = structure.nat

    if not selective_atoms:
        selective_atoms = list(range(N))

    # First construct full matrix of all kernel vectors in parallel,
    # then predict on atoms in parallel

    kernel_mat = get_kernel_mat(
        structure,
        gp.name,
        gp.kernel,
        gp.hyps,
        gp.cutoffs,
        gp.hyps_mask,
        selective_atoms,
    )

    Npred = len(selective_atoms)
    Ntot = 3 * Npred
    forces = np.zeros(shape=(N, 3))
    stds = np.zeros(shape=(N, 3))

    selective_forces = np.zeros((Npred, 3))
    selective_stds = np.zeros((Npred, 3))

    forces.fill(skipped_atom_value)
    stds.fill(skipped_atom_value)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ranks = np.arange(size)
    local_starts = np.floor(Ntot * ranks / size).astype(np.int32)
    local_ends = np.floor(Ntot * (ranks + 1) / size).astype(np.int32)
    local_sizes = local_ends - local_starts

    old_atom = -13

    for x in range(local_starts[rank], local_ends[rank]):
        atom = x // 3
        dim = x - atom * 3
        if old_atom != atom:
            old_atom = atom
            x_t = AtomicEnvironment(structure, selective_atoms[atom], gp.cutoffs)
        force, var = gp.predict_on_kernel_vec(
            kernel_mat[atom, dim].reshape(-1), x_t, dim + 1
        )
        selective_forces[atom][dim] = float(force)
        selective_stds[atom][dim] = float(np.sqrt(np.absolute(var)))

    comm.Allgatherv(
        MPI.IN_PLACE, [selective_forces, local_sizes, local_starts, MPI.DOUBLE]
    )
    comm.Allgatherv(
        MPI.IN_PLACE, [selective_stds, local_sizes, local_starts, MPI.DOUBLE]
    )

    for i in range(Npred):
        forces[selective_atoms[i]] = selective_forces[i]
        stds[selective_atoms[i]] = selective_stds[i]

    if write_to_structure:
        structure.forces = forces
        structure.stds = stds

    return forces, stds


def predict_on_structure_par(
    structure: Structure,
    gp: GaussianProcess,
    n_cpus: int = None,
    write_to_structure: bool = True,
    selective_atoms: List[int] = None,
    skipped_atom_value=0,
) -> ("np.ndarray", "np.ndarray"):
    """
    Return the forces/std. dev. uncertainty associated with each
    individual atom in a structure. Forces are stored directly to the
    structure and are also returned.

    :param structure: FLARE structure to obtain forces for, with N atoms
    :param gp: Gaussian Process model
    :param n_cpus: Number of cores to parallelize over
    :param write_to_structure: Write results to structure's forces,
                            std attributes
    :param selective_atoms: Only predict on these atoms; e.g. [0,1,2] will
                                only predict and return for those atoms
    :param skipped_atom_value: What value to use for atoms that are skipped.
            Defaults to 0 but other options could be e.g. NaN. Will NOT
            write this to the structure if write_to_structure is True.
    :return: N x 3 array of forces, N x 3 array of uncertainties
    :rtype: (np.ndarray, np.ndarray)
    """
    # Just work in serial in the number of cpus is 1
    if n_cpus is 1:
        return predict_on_structure(
            structure=structure,
            gp=gp,
            n_cpus=n_cpus,
            write_to_structure=write_to_structure,
            selective_atoms=selective_atoms,
            skipped_atom_value=skipped_atom_value,
        )

    forces = np.zeros(shape=(structure.nat, 3))
    stds = np.zeros(shape=(structure.nat, 3))

    if selective_atoms:
        forces.fill(skipped_atom_value)
        stds.fill(skipped_atom_value)
    else:
        selective_atoms = []

    # Automatically detect number of cpus available
    if n_cpus is None:
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=n_cpus)

    # Parallelize over atoms in structure
    results = []
    for atom in range(structure.nat):
        # If selective atoms is on, skip ones that was skipped
        if atom not in selective_atoms and selective_atoms:
            # Keep length of results equal to nat
            results.append(None)
            continue
        results.append(pool.apply_async(predict_on_atom, args=[(structure, atom, gp)]))
    pool.close()
    pool.join()

    for i in range(structure.nat):
        if i not in selective_atoms:
            continue
        r = results[i].get()
        forces[i] = r[0]
        stds[i] = r[1]
        if write_to_structure:
            structure.forces[i] = r[0]
            structure.stds[i] = r[1]

    return forces, stds


def predict_on_structure_en(
    structure: Structure,
    gp: GaussianProcess,
    n_cpus: int = None,
    write_to_structure: bool = True,
    selective_atoms: List[int] = None,
    skipped_atom_value=0,
) -> ("np.ndarray", "np.ndarray", "np.ndarray"):
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
    N = structure.nat

    if not selective_atoms:
        selective_atoms = list(range(N))

    Npred = len(selective_atoms)
    # Set up local energy array
    local_energies = np.zeros(structure.nat)
    selective_energies = np.zeros(Npred)

    forces = np.zeros(shape=(structure.nat, 3))
    stds = np.zeros(shape=(structure.nat, 3))

    forces.fill(skipped_atom_value)
    stds.fill(skipped_atom_value)
    local_energies.fill(skipped_atom_value)

    forces, stds = predict_on_structure(
        structure,
        gp,
        write_to_structure,
        selective_atoms=selective_atoms,
        skipped_atom_value=skipped_atom_value,
    )

    kernel_mat = get_en_kernel_mat(
        structure,
        gp.name,
        gp.energy_force_kernel,
        gp.hyps,
        gp.cutoffs,
        gp.hyps_mask,
        selective_atoms,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = structure.nat

    ranks = np.arange(size)
    local_starts = np.floor(Npred * ranks / size).astype(np.int32)
    local_ends = np.floor(Npred * (ranks + 1) / size).astype(np.int32)
    local_sizes = local_ends - local_starts

    # Loop through atoms in structure and predict forces, uncertainties,
    # and energies
    for n in range(local_starts[rank], local_ends[rank]):
        selective_energies[n] = gp.predict_en_on_kern_vec(kernel_mat[n].reshape(-1))

    comm.Allgatherv(
        MPI.IN_PLACE, [selective_energies, local_sizes, local_starts, MPI.DOUBLE]
    )

    for i in range(Npred):
        local_energies[selective_atoms[i]] = selective_energies[i]

    return forces, stds, local_energies


def predict_on_structure_par_en(
    structure: Structure,
    gp: GaussianProcess,
    n_cpus: int = None,
    write_to_structure: bool = True,
    selective_atoms: List[int] = None,
    skipped_atom_value=0,
) -> ("np.ndarray", "np.ndarray", "np.ndarray"):
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

    local_energies = np.zeros(structure.nat)
    forces = np.zeros(shape=(structure.nat, 3))
    stds = np.zeros(shape=(structure.nat, 3))

    if selective_atoms:
        forces.fill(skipped_atom_value)
        stds.fill(skipped_atom_value)
        local_energies.fill(skipped_atom_value)
    else:
        selective_atoms = []

    # Work in serial if the number of cpus is 1
    if n_cpus is 1:
        return predict_on_structure_en(
            structure,
            gp,
            write_to_structure=write_to_structure,
            selective_atoms=selective_atoms,
            skipped_atom_value=skipped_atom_value,
        )

    if n_cpus is None:
        pool = mp.Pool(processes=mp.cpu_count())
    else:
        pool = mp.Pool(processes=n_cpus)

    results = []
    # Parallelize over atoms in structure
    for atom_i in range(structure.nat):

        if atom_i not in selective_atoms and selective_atoms:
            results.append(None)
            continue

        results.append(
            pool.apply_async(predict_on_atom_en, args=[(structure, atom_i, gp)])
        )
    pool.close()
    pool.join()

    # Compile results
    for i in range(structure.nat):

        if i not in selective_atoms and selective_atoms:
            continue

        r = results[i].get()
        forces[i][:] = r[0]
        stds[i][:] = r[1]
        local_energies[i] = r[2]

        if write_to_structure:
            structure.forces[i] = forces[i]
            structure.stds[i] = stds[i]

    return forces, stds, local_energies
