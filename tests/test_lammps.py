import pytest
import os, shutil
import numpy as np
from flare.utils.element_coder import _Z_to_mass, _Z_to_element, _element_to_Z
from flare.ase.atoms import FLARE_Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read, write

from .get_sgp import get_sgp_calc, get_random_atoms

n_species_list = [1, 2]
n_desc_types = [1, 2]
power_list = [1, 2]
rootdir = os.getcwd()

@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found in environment: Please install LAMMPS and set the "
        "$lmp environment variable to point to the executatble."
    ),
)
@pytest.mark.parametrize("n_species", n_species_list)
@pytest.mark.parametrize("n_types", n_desc_types)
@pytest.mark.parametrize("power", power_list)
def test_write_potential(n_species, n_types, power):
    """Test the flare_pp pair style."""

    if n_species > n_types:
        pytest.skip()

    if (power == 1) and ("kokkos" in os.environ.get("lmp")):
        pytest.skip()

    # Write potential file.
    sgp_model = get_sgp_calc(n_types, power)
    potential_name = f"LJ_{n_species}_{n_types}_{power}.txt"
    contributor = "Jon"
    kernel_index = 0
    sgp_model.gp_model.sparse_gp.write_mapping_coefficients(
        potential_name, contributor, kernel_index)

    # Generate random testing structure
    if n_species == 1:
        numbers = [6, 6]
        species = ["C"]
    elif n_species == 2:
        numbers = [6, 8]
        species = ["C", "O"]
    test_structure = get_random_atoms(a=2.0, sc_size=2, numbers=numbers)
    test_structure.calc = sgp_model

    # Predict with SGP.
    energy = test_structure.get_potential_energy()
    forces = test_structure.get_forces()
    stress = test_structure.get_stress()

    # Set up LAMMPS calculator.
    parameters = {
        "command": os.environ.get("lmp"),  # set up executable for ASE
        "newton": "on",
        "pair_style": "flare",
        "pair_coeff": [f"* * {potential_name}"],
    }

    lmp_calc = LAMMPS(tmp_dir="./tmp/", parameters=parameters,
                      files=[potential_name], specorder=species)

    print("built lmp_calc")
    # Predict with LAMMPS.
    test_structure.calc = lmp_calc
    energy_lmp = test_structure.get_potential_energy()
    forces_lmp = test_structure.get_forces()
    stress_lmp = test_structure.get_stress()

    thresh = 1e-6
    print(energy, energy_lmp)
    assert(np.abs(energy - energy_lmp) < thresh)
    print(forces)
    print(forces_lmp)
    assert(np.max(np.abs(forces - forces_lmp)) < thresh)
    assert(np.max(np.abs(stress - stress_lmp)) < thresh)

    # Remove files.
    os.remove(potential_name)
    os.system("rm -r tmp")


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
@pytest.mark.parametrize("n_species", n_species_list)
@pytest.mark.parametrize("n_types", n_desc_types)
@pytest.mark.parametrize("use_map", [False, True])
@pytest.mark.parametrize("power", power_list)
def test_lammps_uncertainty(n_species, n_types, use_map, power):
    if n_species > n_types:
        pytest.skip()

    if (power == 1) and ("kokkos" in os.environ.get("lmp")):
        pytest.skip()

    os.chdir(rootdir)
    lmp_command = os.environ.get("lmp")

    # get sgp & dump coefficient files
    sgp_model = get_sgp_calc(n_types, power)
    contributor = "YX"
    kernel_index = 0

    potential_file = f"LJ_{n_species}_{n_types}_{power}.txt"
    sgp_model.gp_model.write_mapping_coefficients(
        potential_file, contributor, kernel_index)

    if use_map:
        varmap_file = f"varmap_{n_species}_{n_types}.txt"
        sgp_model.gp_model.write_varmap_coefficients(
            varmap_file, contributor, kernel_index)
        coeff_str = varmap_file
    else:
        L_inv_file = f"Linv_{n_species}_{n_types}.txt"
        sgp_model.gp_model.sparse_gp.write_L_inverse(
            L_inv_file, contributor)
    
        sparse_desc_file = f"sparse_desc_{n_species}_{n_types}.txt"
        sgp_model.gp_model.sparse_gp.write_sparse_descriptors(
            sparse_desc_file, contributor)
        coeff_str = f"{L_inv_file} {sparse_desc_file}"

    # Generate random testing structure
    if n_species == 1:
        numbers = [6, 6]
        species = ["C"]
        mass_str = "mass 1 12"
    elif n_species == 2:
        numbers = [6, 8]
        species = ["C", "O"]
        mass_str = "mass 1 12\nmass 2 16"
    test_atoms = get_random_atoms(a=2.0, sc_size=2, numbers=numbers)

    # compute uncertainty 
    in_lmp = f"""
atom_style atomic 
units metal
boundary p p p 
atom_modify sort 0 0.0 

read_data data.lammps 

### interactions
pair_style flare 
pair_coeff * * {potential_file}
{mass_str}

### run
fix fix_nve all nve
compute unc all flare/std/atom {coeff_str}
dump dump_all all custom 1 traj.lammps id type x y z vx vy vz fx fy fz c_unc 
thermo_style custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms
thermo_modify flush yes format float %23.16g
thermo 1
run 0
"""
    if "tmp" not in os.listdir():
        os.mkdir("tmp")

    os.chdir("tmp")
    write("data.lammps", test_atoms, format="lammps-data")
    with open("in.lammps", "w") as f:
        f.write(in_lmp)
    shutil.copyfile(f"../{potential_file}", f"./{potential_file}")
    if use_map:
        shutil.copyfile(f"../{varmap_file}", f"./{varmap_file}")
    else:
        shutil.copyfile(f"../{L_inv_file}", f"./{L_inv_file}")
        shutil.copyfile(f"../{sparse_desc_file}", f"./{sparse_desc_file}")
    os.system(f"{lmp_command} < in.lammps > log.lammps")
    unc_atoms = read("traj.lammps", format="lammps-dump-text")
    lmp_stds = unc_atoms.get_array(f"c_unc")

    # Test mapped variance (need to use sgp_var)
    test_atoms.calc = None
    test_atoms = FLARE_Atoms.from_ase_atoms(test_atoms)
    test_atoms.calc = sgp_model
    if use_map:
        test_atoms.calc.gp_model.sparse_gp = sgp_model.gp_model.sgp_var
    test_atoms.calc.reset()
    sgp_stds = test_atoms.calc.get_uncertainties(test_atoms)
    print(sgp_stds)
    print(lmp_stds)
    assert np.allclose(sgp_stds[:,0], lmp_stds.squeeze(), rtol=1e-3)

    os.chdir("..")
    os.system("rm -r tmp *.txt")
