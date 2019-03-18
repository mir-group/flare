import numpy as np
import sys
import qe_parsers, analyze_gp, eam, analyze_md, otf_parser_v2
from scipy.optimize import minimize
import time
import copy
sys.path.append('../otf_engine')
import gp, env, struc, kernels, otf, md, md_run


def get_structure_from_dataset(filename, snap):
    box_test = np.load(file_name + 'box.npy')
    coord_test = np.load(file_name + 'coord.npy')
    force_test = np.load(file_name + 'force.npy')
    energy_test = np.load(file_name + 'energy.npy')

    positions = coord_test[snap].reshape(-1, 3)
    forces = force_test[snap].reshape(-1, 3)
    cell = box_test[snap].reshape(3, 3)
    species = ['Cu'] * positions.shape[0]

    struc_curr = struc.Structure(cell, species, positions)

    return struc_curr, forces


def predict_forces_on_structure(gp_model, structure, forces, cutoffs):
    noa = len(structure.positions)

    predictions = np.zeros([noa, 3])
    variances = np.zeros([noa, 3])

    for atom in range(noa):
        env_curr = env.AtomicEnvironment(structure, atom, cutoffs)

        for n in range(3):
            d = n+1
            comp_pred, comp_var = gp_model.predict(env_curr, d)
            predictions[atom, n] = comp_pred
            variances[atom, n] = comp_var

    # mean absolute error
    MAE = np.mean(np.abs(forces-predictions))

    # mean absolute std
    stds = np.sqrt(variances)
    MAS = np.mean(stds)

    return predictions, variances, forces, MAE, MAS


def predict_forces_on_snaps(gp_model, filename, snaps, cutoffs):
    preds = np.array([])
    frcs = np.array([])
    for snap in snaps:
        struc_curr, forces = get_structure_from_dataset(filename, snap)
        predictions, variances, forces, MAE, MAS = \
            predict_forces_on_structure(gp_model, struc_curr, forces, cutoffs)
        preds = np.append(preds, predictions)
        frcs = np.append(frcs, forces)
    return preds, frcs

# load copper training set (1/9)
box_test = np.load('/Users/jonpvandermause/Downloads/Cu/set.000/box.npy')
coord_test = np.load('/Users/jonpvandermause/Downloads/Cu/set.000/coord.npy')
force_test = np.load('/Users/jonpvandermause/Downloads/Cu/set.000/force.npy')
energy_test = np.load('/Users/jonpvandermause/Downloads/Cu/set.000/energy.npy')

# initialize gp
kernel = kernels.three_body
kernel_grad = kernels.three_body_grad
hyps = np.array([1, 1, 1])
cutoffs = np.array([4.2, 4.2])
opt_algorithm = 'BFGS'
maxiter = 50

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                              opt_algorithm=opt_algorithm, maxiter=maxiter)

snaps = [100, 125, 150, 175, 200, 225, 250, 275, 300]
atoms = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

for snap in snaps:
    positions = coord_test[snap].reshape(-1, 3)
    forces = force_test[snap].reshape(-1, 3)
    cell = box_test[snap].reshape(3, 3)
    species = ['Cu'] * positions.shape[0]

    struc_curr = struc.Structure(cell, species, positions)

    gp_model.update_db(struc_curr, forces, atoms)

gp_model.hyps = np.array([0.00658963, 1.16094213, 0.10436963])
gp_model.set_L_alpha()

snaps = [100, 200, 300]
file_name = '/Users/jonpvandermause/Downloads/Cu/set.009/'

predictions, forces = \
    predict_forces_on_snaps(gp_model, file_name, snaps, cutoffs)

print(np.sqrt(np.mean((predictions - forces)**2)))
