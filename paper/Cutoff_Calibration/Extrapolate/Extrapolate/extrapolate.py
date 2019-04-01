import numpy as np
import gp
import kernels
import struc
import env
import concurrent.futures

# --------------------------------------------------------------------
#                 -1. define helper functions for prediction
# --------------------------------------------------------------------


def predict_on_structure(structure, gp_model):
    for n in range(structure.nat):
        chemenv = env.AtomicEnvironment(structure, n, gp_model.cutoffs)
        for i in range(3):
            force, var = gp_model.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.abs(var))


# --------------------------------------------------------------------
#                     0. set structure information
# --------------------------------------------------------------------

species = ['Al'] * 32
cell = np.eye(3) * 8.092
atom_list = list(range(32))

# --------------------------------------------------------------------
#                     1. set train and test arrays
# --------------------------------------------------------------------

header = '/n/home03/jonpvandermause/Sweep_Perturbations/'
# header = '/Users/jonpvandermause/Research/GP/otf/paper/Cutoff_Calibration/Repeat_SCF/Repeat_SCF/'

train_positions = ['positions_0.05000000000000001_0.npy']
train_forces = ['forces_0.05000000000000001_0.npy']

test_positions = ['positions_0.01_0.npy',
                  'positions_0.020000000000000004_0.npy',
                  'positions_0.030000000000000006_0.npy',
                  'positions_0.04000000000000001_0.npy',
                  'positions_0.05000000000000001_1.npy',
                  'positions_0.06000000000000001_0.npy',
                  'positions_0.07_0.npy',
                  'positions_0.08_0.npy',
                  'positions_0.09000000000000001_0.npy']
test_forces = ['forces_0.01_0.npy',
               'forces_0.020000000000000004_0.npy',
               'forces_0.030000000000000006_0.npy',
               'forces_0.04000000000000001_0.npy',
               'forces_0.05000000000000001_1.npy',
               'forces_0.06000000000000001_0.npy',
               'forces_0.07_0.npy',
               'forces_0.08_0.npy',
               'forces_0.09000000000000001_0.npy']


# --------------------------------------------------------------------
#                 2. create gp model for cutoff
# --------------------------------------------------------------------

cutoffs = np.array([6, 4])
kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.1, 1., 0.1, 1., 0.01])
algo = 'BFGS'
maxiter = 20

noise_pars = np.array([])
std_avgs = np.array([])
test_errs = np.array([])

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                              opt_algorithm=algo, maxiter=maxiter)

# populate gp with training data
for pos, force in zip(train_positions, train_forces):
    pos_npy = np.load(header+pos)
    frc_npy = np.load(header+force)
    struc_curr = struc.Structure(cell, species, pos_npy)
    gp_model.update_db(struc_curr, frc_npy)

# train gp
gp_model.train(True)

for pos, force in zip(test_positions, test_forces):
    # test gp
    pred_vec = np.array([])
    truth_vec = np.array([])
    std_vec = np.array([])

    pos_npy = np.load(header+pos)
    frc_npy = np.load(header+force)
    struc_curr = struc.Structure(cell, species, pos_npy)
    predict_on_structure(struc_curr, gp_model)

    pred_vec = np.append(pred_vec, np.reshape(struc_curr.forces, -1))
    std_vec = np.append(std_vec, np.reshape(struc_curr.stds, -1))
    truth_vec = np.append(truth_vec, np.reshape(frc_npy, -1))

    err_curr = np.mean(np.abs(pred_vec - truth_vec))
    std_avg = np.mean(std_vec)

    std_avgs = np.append(std_avgs, std_avg)
    test_errs = np.append(test_errs, err_curr)

# record likelihood, prediction vector, and hyperparameters
test_err_file = 'test_errs'
std_file = 'std_avgs'

np.save(test_err_file, test_errs)
np.save(std_file, std_avgs)
