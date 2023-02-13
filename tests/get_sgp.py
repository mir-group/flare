import numpy as np
from flare.bffs.sgp._C_flare import NormalizedDotProduct, DotProduct, B1, B2, B3
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.parallel_sgp import ParSGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.atoms import FLARE_Atoms
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import make_supercell

# Define kernels.
sigma = 1.0
power = 2
kernel2 = NormalizedDotProduct(sigma, power)

sigma = 2.0
power = 2
kernel3 = NormalizedDotProduct(sigma, power)

# Define calculators.
species_map = {6: 0, 8: 1}

cutoff = 5.0
cutoff_function = "quadratic"
radial_basis = "chebyshev"
radial_hyps = [0.0, cutoff]
cutoff_hyps = []

settings = [len(species_map), 4, 3]
calc2 = B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

settings = [len(species_map), 2, 2]
calc3 = B3(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

# Define kernel.
sigma = 2.0
power = 1.0
dotprod_kernel = DotProduct(sigma, power)
normdotprod_kernel = NormalizedDotProduct(sigma, power)

# Define remaining parameters for the SGP wrapper.
sigma_e = 0.3
sigma_f = 0.2
sigma_s = 0.1
species_map = {6: 0, 8: 1}
single_atom_energies = {0: -5, 1: -6}
variance_type = "local"
max_iterations = 20
opt_method = "BFGS"
bounds = [(None, None), (sigma_e, None), (None, None), (None, None)]


def get_random_atoms(a=2.0, sc_size=2, numbers=[6, 8], set_seed: int = None):

    """Create a random structure."""

    if set_seed:
        np.random.seed(set_seed)

    cell = np.eye(3) * a
    positions = np.array([[0, 0, 0], np.random.rand(3)])
    unit_cell = Atoms(cell=cell, positions=positions, numbers=numbers, pbc=True)
    multiplier = np.identity(3) * sc_size
    atoms = make_supercell(unit_cell, multiplier)
    atoms.positions += (2 * np.random.rand(len(atoms), 3) - 1) * 0.1
    flare_atoms = FLARE_Atoms.from_ase_atoms(atoms)

    calc = SinglePointCalculator(flare_atoms)
    calc.results["energy"] = np.random.rand()
    calc.results["forces"] = np.random.rand(len(atoms), 3)
    calc.results["stress"] = np.random.rand(6)
    flare_atoms.calc = calc
 
    return flare_atoms


def get_isolated_atoms(numbers=[6, 8]):

    """Create a random structure."""

    a = 30.0
    cell = np.eye(3) * a
    positions = np.array([[0, 0, 0], [1, 1, 1], [a / 2, a / 2, a / 2]])
    if 8 in numbers:
        numbers = [6, 8, 8]
    else:
        numbers = [6, 6, 6]
    unit_cell = Atoms(cell=cell, positions=positions, numbers=numbers, pbc=True)
    atoms = unit_cell
    flare_atoms = FLARE_Atoms.from_ase_atoms(atoms)

    return flare_atoms


def get_empty_sgp(n_types=2, power=2, multiple_cutoff=False, kernel_type="NormalizedDotProduct", bk=2):
    if kernel_type == "NormalizedDotProduct":
        kernel = normdotprod_kernel
    elif kernel_type == "DotProduct":
        kernel = dotprod_kernel

    kernel.power = power

    # Define B2 calculator.
    cutoff = 5.0
    cutoff_function = "quadratic"
    radial_basis = "chebyshev"
    radial_hyps = [0.0, cutoff]
    cutoff_hyps = []
    cutoff_matrix = cutoff * np.ones((n_types, n_types))
    if multiple_cutoff:
        cutoff_matrix += np.eye(n_types) - 1

    descriptor_settings = [n_types, 3, 2]
    if bk == 1:
        calc = B1(
            radial_basis,
            cutoff_function,
            radial_hyps,
            cutoff_hyps,
            descriptor_settings,
            #cutoff_matrix,
        )
    elif bk == 2:
        calc = B2(
            radial_basis,
            cutoff_function,
            radial_hyps,
            cutoff_hyps,
            descriptor_settings,
            cutoff_matrix,
        )
    elif bk == 3:
        calc = B3(
            radial_basis,
            cutoff_function,
            radial_hyps,
            cutoff_hyps,
            descriptor_settings,
            #cutoff_matrix,
        )

    empty_sgp = SGP_Wrapper(
        [kernel],
        [calc],
        cutoff,
        sigma_e,
        sigma_f,
        sigma_s,
        species_map,
        single_atom_energies=single_atom_energies,
        variance_type=variance_type,
        opt_method=opt_method,
        bounds=bounds,
        max_iterations=max_iterations,
    )

    return empty_sgp


def get_updated_sgp(n_types=2, power=2, multiple_cutoff=False, kernel_type="NormalizedDotProduct", bk=2):
    if n_types == 1:
        numbers = [6, 6]
    elif n_types == 2:
        numbers = [6, 8]

    sgp = get_empty_sgp(n_types, power, multiple_cutoff, kernel_type, bk)

    # add a random structure to the training set
    training_structure = get_random_atoms(numbers=numbers, sc_size=2)
    training_structure.calc = LennardJones()

    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()

    sgp.update_db(
        training_structure,
        forces,
        custom_range=(1, 2, 3, 4, 5),
        energy=energy,
        stress=stress,
        mode="specific",
        rel_e_noise=0.1,
        rel_f_noise=0.2,
        rel_s_noise=0.1,
    )

    # add an isolated atom to the training data
    training_structure = get_isolated_atoms(numbers=numbers)
    training_structure.calc = LennardJones()

    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()

    sgp.update_db(
        training_structure,
        forces,
        custom_range=(0,),
        energy=energy,
        stress=stress,
        mode="specific",
    )

    print("sparse_indices", sgp.sparse_gp.sparse_indices)

    return sgp

def get_empty_parsgp():
    empty_sgp = ParSGP_Wrapper(
        [kernel2, kernel3], 
        [calc2, calc3], 
        cutoff, 
        sigma_e, 
        sigma_f, 
        sigma_s, 
        species_map,
        single_atom_energies=single_atom_energies, 
        variance_type=variance_type,
        opt_method=opt_method, 
        bounds=bounds, 
        max_iterations=max_iterations,
    )

    return empty_sgp

def get_empty_multikern_sgp():
    empty_sgp = SGP_Wrapper(
        [kernel2, kernel3], 
        [calc2, calc3], 
        cutoff, 
        sigma_e, 
        sigma_f, 
        sigma_s, 
        species_map,
        single_atom_energies=single_atom_energies, 
        variance_type=variance_type,
        opt_method=opt_method, 
        bounds=bounds, 
        max_iterations=max_iterations,
    )

    return empty_sgp

def get_sgp_calc(n_types=2, power=2, multiple_cutoff=False, kernel_type="NormalizedDotProduct", bk=2):
    sgp = get_updated_sgp(n_types, power, multiple_cutoff, kernel_type, bk)
    sgp_calc = SGP_Calculator(sgp)

    return sgp_calc

def get_training_data():
    # Make random structure.
    sgp = get_empty_multikern_sgp()
    n_frames = 5
    training_strucs = []
    training_sparse_indices = [[] for i in range(len(sgp.descriptor_calculators))]
    sc_size_list = np.random.randint(2, size=n_frames) + 2
    for n in range(n_frames): 
        train_structure = get_random_atoms(a=2.0, sc_size=sc_size_list[n], numbers=list(species_map.keys()))
        n_atoms = len(train_structure)
        training_strucs.append(train_structure)
        for k in range(len(sgp.descriptor_calculators)):
            training_sparse_indices[k].append(np.random.randint(0, n_atoms, n_atoms // 2).tolist())
    return training_strucs, training_sparse_indices
