import os, shutil, re
import numpy as np
from flare import env, struc


def clean():
    for f in os.listdir("./"):
        if re.search("mgp_grids", f):
            shutil.rmtree(f)
        if re.search("kv3", f):
            os.rmdir(f)
        if 'tmp' in f:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
        if '.mgp' in f:
            os.remove(f)


def compare_triplet(mgp_model, gp_model, atom_env):
    spcs, comp_r, comp_xyz = mgp_model.get_arrays(atom_env)
    for i, spc in enumerate(spcs):
        lengths = np.array(comp_r[i])
        xyzs = np.array(comp_xyz[i])

        print('compare triplet spc, lengths, xyz', spc)
        print(np.hstack([lengths, xyzs]))

        gp_f = []
        gp_e = []
        grid_env = get_grid_env(gp_model, spc, 3)
        for l in range(lengths.shape[0]):
            r1, r2, r12 = lengths[l, :]
            grid_env = get_triplet_env(r1, r2, r12, grid_env)
            gp_pred = np.array([gp_model.predict(grid_env, d+1) for d in range(3)]).T
            gp_en, _ = gp_model.predict_local_energy_and_var(grid_env)
            gp_f.append(gp_pred[0])
            gp_e.append(gp_en)
        gp_force = np.sum(gp_f, axis=0)
        gp_energy = np.sum(gp_e, axis=0)
        print('gp_e', gp_e)
        print('gp_f')
        print(gp_f)

        map_ind = mgp_model.find_map_index(spc)
        xyzs = np.zeros_like(xyzs)
        xyzs[:, 0] = np.ones_like(xyzs[:, 0])
        f, _, _, e = mgp_model.maps[map_ind].predict(lengths, xyzs,
            mgp_model.map_force, mean_only=True)

        assert np.allclose(gp_force, f, rtol=1e-2)
        if not mgp_model.map_force:
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
    positions = [[(i+1)/(bodies+1)*0.1, 0, 0]
                 for i in range(bodies)]
    grid_struc = struc.Structure(big_cell, species, positions)
    grid_env = env.AtomicEnvironment(grid_struc, 0, GP.cutoffs,
        cutoffs_mask=GP.hyps_mask)

    return grid_env


def diag_var():
