import pytest
import os, shutil
import numpy as np
from flare.atoms import FLARE_Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read, write

from .get_sgp import get_sgp_calc, get_random_atoms, get_isolated_atoms

n_species_list = [1, 2]
n_desc_types = [1, 2]
power_list = [1, 2]
struc_list = ["random", "isolated"]
rootdir = os.getcwd()
n_cpus_list = [1]  # [1, 2]


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
@pytest.mark.parametrize("struc", struc_list)
@pytest.mark.parametrize("multicut", [False, True])
@pytest.mark.parametrize("n_cpus", n_cpus_list)
def test_write_potential(n_species, n_types, power, struc, multicut, n_cpus):
    """Test the flare pair style."""

    if n_species > n_types:
        pytest.skip()

    if (power == 1) and ("kokkos" in os.environ.get("lmp")):
        pytest.skip()

    # Write potential file.
    sgp_model = get_sgp_calc(n_types, power, multicut)
    potential_name = f"LJ_{n_species}_{n_types}_{power}.txt"
    contributor = "Jon"
    kernel_index = 0
    sgp_model.gp_model.sparse_gp.write_mapping_coefficients(
        potential_name, contributor, kernel_index
    )

    # Generate random testing structure
    if n_species == 1:
        numbers = [6, 6]
        species = ["C"]
    elif n_species == 2:
        numbers = [6, 8]
        species = ["C", "O"]

    if struc == "random":
        test_structure = get_random_atoms(a=2.0, sc_size=2, numbers=numbers)
    elif struc == "isolated":
        test_structure = get_isolated_atoms(numbers=numbers)
    test_structure.calc = sgp_model

    # Predict with SGP.
    energy = test_structure.get_potential_energy()
    forces = test_structure.get_forces()
    stress = test_structure.get_stress()

    # Set up LAMMPS calculator.
    lmp_command = os.environ.get("lmp")
    if (n_cpus > 1) and ("mpirun" not in lmp_command) and ("kokkos" not in lmp_command):
        lmp_command = f"mpirun -np {n_cpus} {lmp_command}"

    print(lmp_command)
    parameters = {
        "command": lmp_command,  # set up executable for ASE
        "newton": "on",
        "pair_style": "flare",
        "pair_coeff": [f"* * {potential_name}"],
        "timestep": "0.001\ndump_modify dump_all sort id",
    }

    lmp_calc = LAMMPS(
        tmp_dir="./tmp/",
        parameters=parameters,
        files=[potential_name],
        specorder=species,
    )

    print("built lmp_calc")
    # Predict with LAMMPS.
    test_structure.calc = lmp_calc
    energy_lmp = test_structure.get_potential_energy()
    forces_lmp = test_structure.get_forces()
    stress_lmp = test_structure.get_stress()

    # add back single_atom_energies to lammps energy
    if sgp_model.gp_model.single_atom_energies is not None:
        for spec in test_structure.numbers:
            coded_spec = sgp_model.gp_model.species_map[spec]
            energy_lmp += sgp_model.gp_model.single_atom_energies[coded_spec]

    thresh = 1e-6
    print(energy, energy_lmp)
    assert np.allclose(energy, energy_lmp, atol=thresh)
    # print(forces, forces_lmp)
    assert np.allclose(forces, forces_lmp, atol=thresh)
    assert np.allclose(stress, stress_lmp, atol=thresh)

    # Remove files.
    os.remove(potential_name)
    os.system("rm -r tmp")


import os
import numpy as np
from copy import deepcopy
from ase import Atom
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
from flare.md.lammps import LAMMPS_MOD, LAMMPS_MD, get_kinetic_stress


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
@pytest.mark.parametrize("struc", struc_list)
@pytest.mark.parametrize("multicut", [False, True])
@pytest.mark.parametrize("n_cpus", n_cpus_list)
@pytest.mark.parametrize("kernel_type", ["NormalizedDotProduct", "DotProduct"])
def test_lammps_uncertainty(
    n_species, n_types, use_map, power, struc, multicut, n_cpus, kernel_type,
):
    if n_species > n_types:
        pytest.skip()

    if (power == 1 or kernel_type=="DotProduct") and ("kokkos" in os.environ.get("lmp")):
        pytest.skip()

    os.chdir(rootdir)
    # Set up LAMMPS calculator.
    lmp_command = os.environ.get("lmp")
    if (n_cpus > 1) and ("mpirun" not in lmp_command) and ("kokkos" not in lmp_command):
        lmp_command = f"mpirun -np {n_cpus} {lmp_command}"
    print(lmp_command)

    # get sgp & dump coefficient files
    sgp_model = get_sgp_calc(n_types, power, multicut, kernel_type)
    contributor = "YX"
    kernel_index = 0

    potential_file = f"LJ_{n_species}_{n_types}_{power}.txt"
    sgp_model.gp_model.write_mapping_coefficients(
        potential_file, contributor, kernel_index
    )

    if use_map:
        varmap_file = f"varmap_{n_species}_{n_types}.txt"
        sgp_model.gp_model.write_varmap_coefficients(
            varmap_file, contributor, kernel_index
        )
        coeff_str = varmap_file
    else:
        L_inv_file = f"Linv_{n_species}_{n_types}.txt"
        sgp_model.gp_model.sparse_gp.write_L_inverse(L_inv_file, contributor)

        sparse_desc_file = f"sparse_desc_{n_species}_{n_types}.txt"
        sgp_model.gp_model.sparse_gp.write_sparse_descriptors(
            sparse_desc_file, contributor
        )
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

    if struc == "random":
        test_atoms = get_random_atoms(a=2.5, sc_size=2, numbers=numbers)
    elif struc == "isolated":
        test_atoms = get_isolated_atoms(numbers=numbers)

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
dump_modify dump_all sort id
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
    # print(sgp_stds)
    # print(lmp_stds)
    print(sgp_model.gp_model.hyps)
    assert np.allclose(sgp_stds[:, 0], lmp_stds.squeeze(), rtol=2e-3, atol=3e-3)

    os.chdir("..")
    os.system("rm -r tmp *.txt")


def test_lmp_calc():
    Ni = bulk("Ni", cubic=True)
    H = Atom("H", position=Ni.cell.diagonal() / 2)
    NiH = Ni + H

    files = []
    param_dict = {
        "pair_style": "lj/cut 2.5",
        "pair_coeff": ["* * 1 1"],
        "velocity": ["all create 300 12345 dist gaussian rot yes mom yes"],
        "fix": ["1 all nvt temp 300 300 $(100.0*dt)"],
        "dump_period": 1,
        "timestep": 0.001,
        "keep_alive": False,
    }

    ase_lmp_calc = LAMMPS(
        command=os.environ.get("lmp"),
        label="ase",
        files=files,
        keep_tmp_files=True,
        tmp_dir="tmp",
    )
    ase_lmp_calc.set(**param_dict)
    ase_atoms = deepcopy(NiH)
    ase_atoms.calc = ase_lmp_calc
    ase_atoms.calc.calculate(ase_atoms)

    mod_lmp_calc = LAMMPS_MOD(
        command=os.environ.get("lmp"),
        label="mod",
        files=files,
        keep_tmp_files=True,
        tmp_dir="tmp",
    )
    mod_lmp_calc.set(**param_dict)
    mod_atoms = deepcopy(NiH)
    mod_atoms.calc = mod_lmp_calc
    mod_atoms.calc.calculate(mod_atoms, set_atoms=False)
    mod_stress = mod_atoms.get_stress() + get_kinetic_stress(mod_atoms)

    assert np.allclose(
        ase_atoms.get_potential_energy(), mod_atoms.get_potential_energy()
    )
    assert np.allclose(ase_atoms.get_forces(), mod_atoms.get_forces())
    assert np.allclose(ase_atoms.get_stress(), mod_stress)
    os.system("rm -r tmp")
