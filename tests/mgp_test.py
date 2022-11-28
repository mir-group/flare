import os, shutil, re
import numpy as np
from flare.descriptors import env
from flare.atoms import FLARE_Atoms


def clean(prefix="tmp"):
    for f in os.listdir("./"):
        if re.search("mgp_grids", f):
            shutil.rmtree(f)
        if re.search("kv3", f):
            os.rmdir(f)
        if prefix in f:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


def compare_triplet(mgp_model, gp_model, atom_env):
    spcs, comp_r, comp_xyz = mgp_model.get_arrays(atom_env)
    for i, spc in enumerate(spcs):
        lengths = np.array(comp_r[i])
        xyzs = np.array(comp_xyz[i])

        print("compare triplet spc, lengths, xyz", spc)

        gp_f = []
        gp_e = []
        grid_env = get_grid_env(gp_model, spc, 3)
        for l in range(lengths.shape[0]):
            r1, r2, r12 = lengths[l, :]
            grid_env = get_triplet_env(r1, r2, r12, grid_env)
            gp_pred = np.array([gp_model.predict(grid_env, d + 1) for d in range(3)]).T
            gp_en, _ = gp_model.predict_local_energy_and_var(grid_env)
            gp_f.append(gp_pred[0])
            gp_e.append(gp_en)
        gp_force = np.sum(gp_f, axis=0)
        gp_energy = np.sum(gp_e, axis=0)
        print("gp_e", gp_e)
        print("gp_f")
        print(gp_f)

        map_ind = mgp_model.find_map_index(spc)
        xyzs = np.zeros_like(xyzs)
        xyzs[:, 0] = np.ones_like(xyzs[:, 0])
        f, _, _, e = mgp_model.maps[map_ind].predict(lengths, xyzs)

        assert np.allclose(gp_force, f, rtol=1e-2)
        assert np.allclose(gp_energy, e, rtol=1e-2)


def get_triplet_env(r1, r2, r12, grid_env):
    grid_env.bond_array_3 = np.array([[r1, 1, 0, 0], [r2, 0, 0, 0]])
    grid_env.cross_bond_dists = np.array([[0, r12], [r12, 0]])
    print(grid_env.ctype, grid_env.etypes)

    return grid_env


def get_grid_env(GP, species, bodies):
    if isinstance(GP.cutoffs, dict):
        max_cut = np.max(list(GP.cutoffs.values()))
    else:
        max_cut = np.max(GP.cutoffs)
    big_cell = np.eye(3) * 100
    positions = [[(i + 1) / (bodies + 1) * 0.1, 0, 0] for i in range(bodies)]
    grid_struc = FLARE_Atoms(symbols=species, positions=positions, cell=big_cell)
    grid_env = env.AtomicEnvironment(
        grid_struc, 0, GP.cutoffs, cutoffs_mask=GP.hyps_mask
    )

    return grid_env


def predict_struc_diag_var(struc, gp_model):
    variance = np.zeros((struc.nat, 3))
    for atom in range(struc.nat):
        atom_env = env.AtomicEnvironment(struc, atom, gp_model.cutoffs)
        var = predict_atom_diag_var(atom_env, gp_model)
        variance[atom, :] = var
    return variance


def predict_atom_diag_var_2b(atom_env, gp_model, force_kernel):
    bond_array = atom_env.bond_array_2
    ctype = atom_env.ctype

    var = 0
    for m in range(bond_array.shape[0]):
        ri1 = bond_array[m, 0]
        ci1 = bond_array[m, 1:]
        etype1 = atom_env.etypes[m]

        # build up a struc of triplet for prediction
        cell = np.eye(3) * 100
        positions = np.array([np.zeros(3), ri1 * ci1])
        species = np.array([ctype, etype1])
        spc_struc = FLARE_Atoms(symbols=species, positions=positions, cell=cell)
        spc_struc.numbers = np.array(species)
        env12 = env.AtomicEnvironment(spc_struc, 0, gp_model.cutoffs)

        coord = np.copy(env12.bond_array_2[0, 1:])
        # env12.bond_array_2[0, 1:] = np.array([1., 0., 0.])
        if force_kernel:
            v12 = np.zeros(3)
            for d in range(3):
                _, v12[d] = gp_model.predict(env12, d + 1)
            print("v12", np.sqrt(v12), coord)
        else:
            _, v12 = gp_model.predict_local_energy_and_var(env12)

        var += np.sqrt(v12)

    var = var**2
    return var


def predict_atom_diag_var_3b(atom_env, gp_model, force_kernel):
    bond_array = atom_env.bond_array_3
    triplets = atom_env.triplet_counts
    cross_bond_inds = atom_env.cross_bond_inds
    cross_bond_dists = atom_env.cross_bond_dists
    ctype = atom_env.ctype

    var = 0
    pred_dict = {}
    for m in range(bond_array.shape[0]):
        ri1 = bond_array[m, 0]
        ci1 = bond_array[m, 1:]
        etype1 = atom_env.etypes[m]

        for n in range(triplets[m]):
            ind1 = cross_bond_inds[m, m + n + 1]
            ri2 = bond_array[ind1, 0]
            ci2 = bond_array[ind1, 1:]
            etype2 = atom_env.etypes[ind1]

            ri3 = cross_bond_dists[m, m + n + 1]

            # build up a struc of triplet for prediction
            cell = np.eye(3) * 100
            positions = np.array([np.zeros(3), ri1 * ci1, ri2 * ci2])
            species = np.array([ctype, etype1, etype2])
            spc_struc = FLARE_Atoms(symbols=species, positions=positions, cell=cell)
            spc_struc.numbers = np.array(species)
            env12 = env.AtomicEnvironment(spc_struc, 0, gp_model.cutoffs)

            #            env12.bond_array_3[0, 1:] = np.array([1., 0., 0.])
            #            env12.bond_array_3[1, 1:] = np.array([0., 0., 0.])
            if force_kernel:
                v12 = np.zeros(3)
                for d in range(3):
                    _, v12[d] = gp_model.predict(env12, d + 1)
                print("v12", np.sqrt(v12), env12.ctype, env12.etypes)
            else:
                _, v12 = gp_model.predict_local_energy_and_var(env12)

            spc = f"{env12.ctype}_{env12.etypes[0]}_{env12.etypes[1]}"
            if spc in pred_dict:
                pred_dict[spc] += np.sqrt(v12)
            else:
                pred_dict[spc] = np.sqrt(v12)

            var += np.sqrt(v12)
    var = var**2
    print(pred_dict)
    return var


def predict_atom_diag_var(atom_env, gp_model, kernel_name, force_kernel=False):
    print("predict diag var")
    if kernel_name == "twobody":
        return predict_atom_diag_var_2b(atom_env, gp_model, force_kernel)
    elif kernel_name == "threebody":
        return predict_atom_diag_var_3b(atom_env, gp_model, force_kernel)
