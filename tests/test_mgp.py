import numpy as np
import os
import pickle
import pytest
import re
import time
import shutil

from copy import deepcopy
from numpy import allclose, isclose

from flare.descriptors import env
from flare.bffs import gp
from flare.utils.parameters import Parameters
from flare.bffs.mgp import MappedGaussianProcess
from flare.bffs.gp.calculator import FLARE_Calculator
from flare.atoms import FLARE_Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.data import atomic_numbers, atomic_masses

from .fake_gp import get_gp, get_random_structure
from .mgp_test import clean, compare_triplet, predict_atom_diag_var

body_list = ["2", "3"]
multi_list = [True, False]
force_block_only = False
curr_path = os.getcwd()


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
@pytest.fixture(scope="module")
def all_gp():

    allgp_dict = {}
    np.random.seed(123)
    for bodies in body_list:
        for multihyps in multi_list:
            gp_model = get_gp(
                bodies,
                "mc",
                multihyps,
                cellabc=[1.5, 1, 2],
                force_only=force_block_only,
                noa=5,
            )
            gp_model.parallel = True
            gp_model.n_cpus = 2

            allgp_dict[f"{bodies}{multihyps}"] = gp_model

    yield allgp_dict
    del allgp_dict


@pytest.fixture(scope="module")
def all_mgp():

    allmgp_dict = {}
    for bodies in ["2", "3", "2+3"]:
        for multihyps in [False, True]:
            allmgp_dict[f"{bodies}{multihyps}"] = None

    yield allmgp_dict
    del allmgp_dict


@pytest.fixture(scope="module")
def all_lmp():

    all_lmp_dict = {}
    species = ["H", "He"]
    specie_symbol_list = " ".join(species)
    masses = [
        f"{i} {atomic_masses[atomic_numbers[species[i]]]}" for i in range(len(species))
    ]
    parameters = {
        "command": os.environ.get("lmp"),  # set up executable for ASE
        "newton": "off",
        "pair_style": "mgp",
        "mass": masses,
    }

    # set up input params
    for bodies in body_list:
        for multihyps in multi_list:
            # create ASE calc
            label = f"{bodies}{multihyps}"
            files = [f"{label}.mgp"]
            by = "yes" if bodies == "2" else "no"
            ty = "yes" if bodies == "3" else "no"
            parameters["pair_coeff"] = [
                f"* * {label}.mgp {specie_symbol_list} {by} {ty}"
            ]

            lmp_calc = LAMMPS(
                label=label,
                keep_tmp_files=True,
                tmp_dir="./tmp/",
                parameters=parameters,
                files=files,
                specorder=species,
            )
            all_lmp_dict[f"{bodies}{multihyps}"] = lmp_calc

    yield all_lmp_dict
    del all_lmp_dict


@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_init(bodies, multihyps, all_mgp, all_gp):
    """
    test the init function
    """

    clean()

    gp_model = all_gp[f"{bodies}{multihyps}"]

    # grid parameters
    grid_params = {}
    if "2" in bodies:
        grid_params["twobody"] = {"grid_num": [160], "lower_bound": [0.02]}
    if "3" in bodies:
        grid_params["threebody"] = {"grid_num": [31, 32, 33], "lower_bound": [0.02] * 3}

    lammps_location = f"{bodies}{multihyps}"
    data = gp_model.training_statistics

    try:
        mgp_model = MappedGaussianProcess(
            grid_params=grid_params,
            unique_species=data["species"],
            n_cpus=1,
            lmp_file_name=lammps_location,
            var_map="simple",
        )
    except:
        mgp_model = MappedGaussianProcess(
            grid_params=grid_params,
            unique_species=data["species"],
            n_cpus=1,
            lmp_file_name=lammps_location,
            var_map=None,
        )

    all_mgp[f"{bodies}{multihyps}"] = mgp_model


@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_build_map(all_gp, all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """
    gp_model = all_gp[f"{bodies}{multihyps}"]
    mgp_model = all_mgp[f"{bodies}{multihyps}"]
    mgp_model.build_map(gp_model)


#    with open(f'grid_{bodies}_{multihyps}.pickle', 'wb') as f:
#        pickle.dump(mgp_model, f)


@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_write_model(all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """
    mgp_model = all_mgp[f"{bodies}{multihyps}"]
    mgp_model.write_model(f"my_mgp_{bodies}_{multihyps}")

    mgp_model.write_model(f"my_mgp_{bodies}_{multihyps}", format="pickle")

    # Ensure that user is warned when a non-mean_only
    # model is serialized into a Dictionary
    with pytest.warns(Warning):
        mgp_model.var_map = "pca"
        mgp_model.as_dict()

    mgp_model.var_map = "simple"
    mgp_model.as_dict()


@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_load_model(all_mgp, bodies, multihyps):
    """
    test the mapping for mc_simple kernel
    """
    name = f"my_mgp_{bodies}_{multihyps}.json"
    all_mgp[f"{bodies}{multihyps}"] = MappedGaussianProcess.from_file(name)
    os.remove(name)

    name = f"my_mgp_{bodies}_{multihyps}.pickle"
    all_mgp[f"{bodies}{multihyps}"] = MappedGaussianProcess.from_file(name)
    os.remove(name)


@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_cubic_spline(all_gp, all_mgp, bodies, multihyps):
    """
    test the predict for mc_simple kernel
    """

    mgp_model = all_mgp[f"{bodies}{multihyps}"]
    delta = 1e-4

    if "3" in bodies:
        body_name = "threebody"
    elif "2" in bodies:
        body_name = "twobody"

    nmap = len(mgp_model.maps[body_name].maps)
    print("nmap", nmap)
    for i in range(nmap):
        maxvalue = np.max(np.abs(mgp_model.maps[body_name].maps[i].mean.__coeffs__))
        if maxvalue > 0:
            comp_code = mgp_model.maps[body_name].maps[i].species_code

            if "3" in bodies:

                c_pt = np.array([[0.3, 0.4, 0.5]])
                c, cderv = (
                    mgp_model.maps[body_name].maps[i].mean(c_pt, with_derivatives=True)
                )
                cderv = cderv.reshape([-1])

                for j in range(3):
                    a_pt = deepcopy(c_pt)
                    b_pt = deepcopy(c_pt)
                    a_pt[0][j] += delta
                    b_pt[0][j] -= delta
                    a = mgp_model.maps[body_name].maps[i].mean(a_pt)[0]
                    b = mgp_model.maps[body_name].maps[i].mean(b_pt)[0]
                    num_derv = (a - b) / (2 * delta)
                    print("spline", comp_code, num_derv, cderv[j])
                    assert np.isclose(num_derv, cderv[j], rtol=1e-2)

            elif "2" in bodies:
                center = np.sum(mgp_model.maps[body_name].maps[i].bounds) / 2.0
                a_pt = np.array([[center + delta]])
                b_pt = np.array([[center - delta]])
                c_pt = np.array([[center]])
                a = mgp_model.maps[body_name].maps[i].mean(a_pt)[0]
                b = mgp_model.maps[body_name].maps[i].mean(b_pt)[0]
                c, cderv = (
                    mgp_model.maps[body_name].maps[i].mean(c_pt, with_derivatives=True)
                )
                cderv = cderv.reshape([-1])[0]
                num_derv = (a - b) / (2 * delta)
                print("spline", num_derv, cderv)
                assert np.isclose(num_derv, cderv, rtol=1e-2)


@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_predict(all_gp, all_mgp, bodies, multihyps):
    """
    test the predict for mc_simple kernel
    """

    gp_model = all_gp[f"{bodies}{multihyps}"]
    mgp_model = all_mgp[f"{bodies}{multihyps}"]

    # # debug
    # filename = f'grid_{bodies}_{multihyps}.pickle'
    # with open(filename, 'rb') as f:
    #     mgp_model = pickle.load(f)

    nenv = 6
    cell = 1.0 * np.eye(3)
    cutoffs = gp_model.cutoffs
    unique_species = gp_model.training_statistics["species"]
    struc_test, f = get_random_structure(cell, unique_species, nenv)
    test_envi = env.AtomicEnvironment(
        struc_test, 0, cutoffs, cutoffs_mask=gp_model.hyps_mask
    )

    if "2" in bodies:
        kernel_name = "twobody"
    elif "3" in bodies:
        kernel_name = "threebody"
        # compare_triplet(mgp_model.maps['threebody'], gp_model, test_envi)

    mgp_f, mgp_e_var, mgp_s, mgp_e = mgp_model.predict(test_envi)

    assert Parameters.compare_dict(
        gp_model.hyps_mask, mgp_model.maps[kernel_name].hyps_mask
    )

    if multihyps:
        gp_e, gp_e_var = gp_model.predict_local_energy_and_var(test_envi)
        gp_f, gp_f_var = gp_model.predict_force_xyz(test_envi)
    else:
        gp_e, gp_f, gp_s, gp_e_var, _, _ = gp_model.predict_efs(test_envi)
        gp_s = -gp_s[[0, 3, 5, 4, 2, 1]]

        # check stress
        assert np.allclose(mgp_s, gp_s, rtol=1e-2)

    # check mgp is within 2 meV/A of the gp
    print("mgp_en, gp_en", mgp_e, gp_e)
    assert np.allclose(mgp_e, gp_e, rtol=2e-3), (
        f"{bodies} body" f" energy mapping is wrong"
    )

    # check forces
    print("isclose?", mgp_f - gp_f, gp_f)
    assert np.allclose(mgp_f, gp_f, atol=1e-3), f"{bodies} body force mapping is wrong"

    if mgp_model.var_map == "simple":
        print(bodies, multihyps)
        for i in range(struc_test.nat):
            test_envi = env.AtomicEnvironment(
                struc_test, i, cutoffs, cutoffs_mask=gp_model.hyps_mask
            )
            mgp_pred = mgp_model.predict(test_envi)
            mgp_var = mgp_pred[1]
            gp_var = predict_atom_diag_var(test_envi, gp_model, kernel_name)
            print("mgp_var, gp_var", mgp_var, gp_var)
            assert np.allclose(mgp_var, gp_var, rtol=1e-2)

    print("struc_test positions", struc_test.positions, struc_test.symbols)


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
@pytest.mark.parametrize("bodies", body_list)
@pytest.mark.parametrize("multihyps", multi_list)
def test_lmp_predict(all_lmp, all_gp, all_mgp, bodies, multihyps):
    """
    test the lammps implementation
    """

    # pytest.skip()

    prefix = f"{bodies}{multihyps}"

    mgp_model = all_mgp[prefix]
    gp_model = all_gp[prefix]
    lmp_calculator = all_lmp[prefix]
    ase_calculator = FLARE_Calculator(gp_model, mgp_model, par=False, use_mapping=True)

    # create test structure
    np.random.seed(1)
    cell = np.diag(np.array([1, 1, 1])) * 4
    nenv = 10
    unique_species = gp_model.training_statistics["species"]
    cutoffs = gp_model.cutoffs
    struc_test, f = get_random_structure(cell, unique_species, nenv)

    # build ase atom from struc
    ase_atoms_flare = deepcopy(struc_test)
    ase_atoms_flare.calc = ase_calculator

    ase_atoms_lmp = deepcopy(struc_test)
    ase_atoms_lmp.calc = lmp_calculator

    try:
        lmp_en = ase_atoms_lmp.get_potential_energy()
        flare_en = ase_atoms_flare.get_potential_energy()

        lmp_stress = ase_atoms_lmp.get_stress()
        flare_stress = ase_atoms_flare.get_stress()

        lmp_forces = ase_atoms_lmp.get_forces()
        flare_forces = ase_atoms_flare.get_forces()
    except Exception as e:
        os.chdir(curr_path)
        print(e)
        raise e

    os.chdir(curr_path)

    # check that lammps agrees with mgp to within 1 meV/A
    print("energy", lmp_en - flare_en, flare_en)
    assert np.isclose(lmp_en, flare_en, atol=1e-3)
    print("force", lmp_forces - flare_forces, flare_forces)
    assert np.isclose(lmp_forces, flare_forces, atol=1e-3).all()
    print("stress", lmp_stress - flare_stress, flare_stress)
    assert np.isclose(lmp_stress, flare_stress, atol=1e-3).all()

    # check the lmp var
    # mgp_std = np.sqrt(mgp_pred[1])
    # print("isclose? diff:", lammps_stds[atom_num]-mgp_std, "mgp value", mgp_std)
    # assert np.isclose(lammps_stds[atom_num], mgp_std, rtol=1e-2)

    clean(prefix=prefix)
