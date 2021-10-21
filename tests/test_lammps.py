import pytest
import numpy as np
from flare.lammps import lammps_calculator
from flare.struc import Structure
import os

# from test_sparse_gp import sgp_wrapper


# @pytest.mark.skipif(
#     not os.environ.get("lmp", False),
#     reason=(
#         "lmp not found in environment: Please install LAMMPS and set the "
#         "$lmp environment variable to point to the executatble."
#     ),
# )
# def test_lmp_predict(sgp_wrapper):
#     """Test the flare_pp pair style."""

#     pass

#     # # pytest.skip()

#     # prefix = f"{bodies}{multihyps}"

#     # mgp_model = all_mgp[prefix]
#     # gp_model = all_gp[prefix]
#     # lmp_calculator = all_lmp[prefix]
#     # ase_calculator = FLARE_Calculator(gp_model, mgp_model, par=False, use_mapping=True)

#     # # create test structure
#     # np.random.seed(1)
#     # cell = np.diag(np.array([1, 1, 1])) * 4
#     # nenv = 10
#     # unique_species = gp_model.training_statistics["species"]
#     # cutoffs = gp_model.cutoffs
#     # struc_test, f = get_random_structure(cell, unique_species, nenv)

#     # # build ase atom from struc
#     # ase_atoms_flare = struc_test.to_ase_atoms()
#     # ase_atoms_flare = FLARE_Atoms.from_ase_atoms(ase_atoms_flare)
#     # ase_atoms_flare.calc = ase_calculator

#     # ase_atoms_lmp = deepcopy(struc_test).to_ase_atoms()
#     # ase_atoms_lmp.calc = lmp_calculator

#     # try:
#     #     lmp_en = ase_atoms_lmp.get_potential_energy()
#     #     flare_en = ase_atoms_flare.get_potential_energy()

#     #     lmp_stress = ase_atoms_lmp.get_stress()
#     #     flare_stress = ase_atoms_flare.get_stress()

#     #     lmp_forces = ase_atoms_lmp.get_forces()
#     #     flare_forces = ase_atoms_flare.get_forces()
#     # except Exception as e:
#     #     os.chdir(curr_path)
#     #     print(e)
#     #     raise e

#     # os.chdir(curr_path)

#     # # check that lammps agrees with mgp to within 1 meV/A
#     # print("energy", lmp_en - flare_en, flare_en)
#     # assert np.isclose(lmp_en, flare_en, atol=1e-3)
#     # print("force", lmp_forces - flare_forces, flare_forces)
#     # assert np.isclose(lmp_forces, flare_forces, atol=1e-3).all()
#     # print("stress", lmp_stress - flare_stress, flare_stress)
#     # assert np.isclose(lmp_stress, flare_stress, atol=1e-3).all()

#     # # check the lmp var
#     # # mgp_std = np.sqrt(mgp_pred[1])
#     # # print("isclose? diff:", lammps_stds[atom_num]-mgp_std, "mgp value", mgp_std)
#     # # assert np.isclose(lammps_stds[atom_num], mgp_std, rtol=1e-2)

#     # clean(prefix=prefix)
