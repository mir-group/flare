import pytest
import os, shutil
import numpy as np
from flare.utils.element_coder import _Z_to_mass, _Z_to_element, _element_to_Z
from flare.ase.atoms import FLARE_Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.io import read, write

from .get_sgp import get_sgp_calc, get_random_atoms, get_updated_sgp


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found in environment: Please install LAMMPS and set the "
        "$lmp environment variable to point to the executatble."
    ),
)
def test_write_potential():
    """Test the flare_pp pair style."""

    # Write potential file.
    sgp_model = get_sgp_calc()
    potential_name = "LJ.txt"
    contributor = "Jon"
    kernel_index = 0
    sgp_model.gp_model.sparse_gp.write_mapping_coefficients(
        potential_name, contributor, kernel_index)

    # Predict with SGP.
    test_structure = get_random_atoms()
    test_structure.calc = sgp_model
    energy = test_structure.get_potential_energy()
    forces = test_structure.get_forces()
    stress = test_structure.get_stress()

    # Set up LAMMPS calculator.
    species = ["C", "O"]
    parameters = {
        "command": os.environ.get("lmp"),  # set up executable for ASE
        "newton": "on",
        "pair_style": "flare",
        "pair_coeff": ["* * LJ.txt"],
    }

    lmp_calc = LAMMPS(tmp_dir="./tmp/", parameters=parameters,
                      files=[potential_name], specorder=species)

    # Predict with LAMMPS.
    test_structure.calc = lmp_calc
    energy_lmp = test_structure.get_potential_energy()
    forces_lmp = test_structure.get_forces()
    stress_lmp = test_structure.get_stress()

    thresh = 1e-6
    assert(np.abs(energy - energy_lmp) < thresh)
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
def test_lammps_uncertainty():
    lmp_command = os.environ.get("lmp")

    # get sgp & dump coefficient files
    sgp_model = get_sgp_calc()
    contributor = "YX"
    kernel_index = 0
    sgp_model.gp_model.sparse_gp.write_mapping_coefficients(
        "LJ.txt", contributor, kernel_index)
    sgp_model.gp_model.sparse_gp.write_L_inverse(
        "L_inv.txt", contributor)
    sgp_model.gp_model.sparse_gp.write_sparse_descriptors(
        "sparse_desc.txt", contributor)

    # create testing structure
    test_atoms = get_random_atoms(a=2.0, sc_size=1, numbers=[6, 8], set_seed=54321)

    # compute uncertainty 
    in_lmp = """
atom_style atomic 
units metal
boundary p p p 
atom_modify sort 0 0.0 

read_data data.lammps 

### interactions
pair_style flare 
pair_coeff * * LJ.txt
mass 1 1.008000 
mass 2 4.002602 

### run
fix fix_nve all nve
compute unc all flare/std/atom L_inv.txt sparse_desc.txt
dump dump_all all custom 1 traj.lammps id type x y z vx vy vz fx fy fz c_unc 
thermo_style custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms
thermo_modify flush yes format float %23.16g
thermo 1
run 0
"""
    os.chdir("tmp")
    write("data.lammps", test_atoms, format="lammps-data")
    with open("in.lammps", "w") as f:
        f.write(in_lmp)
    shutil.copyfile("../LJ.txt", "./LJ.txt")
    shutil.copyfile("../L_inv.txt", "./L_inv.txt")
    shutil.copyfile("../sparse_desc.txt", "./sparse_desc.txt")
    os.system(f"{lmp_command} < in.lammps > log.lammps")
    unc_atoms = read("traj.lammps", format="lammps-dump-text")
#    sgp_py = get_updated_sgp()
    lmp_stds = unc_atoms.get_array(f"c_unc")

    # Test mapped variance (need to use sgp_var)
    test_atoms.calc = None
    test_atoms = FLARE_Atoms.from_ase_atoms(test_atoms)
#    sgp_calc = get_sgp_calc()
    test_atoms.calc = sgp_model
    #test_atoms.calc.gp_model.sparse_gp = sgp_py.sgp_var
    test_atoms.calc.reset()
    sgp_stds = test_atoms.calc.get_uncertainties(test_atoms)
    print(sgp_stds)
    print(lmp_stds)
    assert np.allclose(sgp_stds[:,0], lmp_stds.squeeze())
