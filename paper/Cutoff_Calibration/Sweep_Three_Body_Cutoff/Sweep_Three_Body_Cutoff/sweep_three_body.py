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

header = '/n/home03/jonpvandermause/Repeat_SCF/'
# header = '/Users/jonpvandermause/Research/GP/otf/paper/Cutoff_Calibration/Repeat_SCF/Repeat_SCF/'

train_positions = ['positions_0.05_0.npy']
train_forces = ['forces_0.05_0.npy']

test_positions = ['positions_0.05_1.npy']
test_forces = ['forces_0.05_1.npy']


# --------------------------------------------------------------------
#                 2. create gp model for cutoff
# --------------------------------------------------------------------

two_body_cutoff = 6
cutoffs = np.arange(3.0, 5.1, 0.5)
kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.1, 1., 0.1, 1., 0.01])
algo = 'BFGS'
maxiter = 20

noise_pars = np.array([])
likes = np.array([])
test_errs = np.array([])

for cutoff in cutoffs:
    cutoffs_curr = np.array([two_body_cutoff, cutoff])

    gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs_curr,
                                  opt_algorithm=algo, maxiter=maxiter)

    # populate gp with training data
    for pos, force in zip(train_positions, train_forces):
        pos_npy = np.load(header+pos)
        frc_npy = np.load(header+force)
        struc_curr = struc.Structure(cell, species, pos_npy)
        gp_model.update_db(struc_curr, frc_npy)

    # train gp
    gp_model.train(True)

    # test gp
    pred_vec = np.array([])
    truth_vec = np.array([])
    for pos, force in zip(test_positions, test_forces):
        pos_npy = np.load(header+pos)
        frc_npy = np.load(header+force)
        struc_curr = struc.Structure(cell, species, pos_npy)
        predict_on_structure(struc_curr, gp_model)
        pred_vec = np.append(pred_vec, np.reshape(struc_curr.forces, -1))
        truth_vec = np.append(truth_vec, np.reshape(frc_npy, -1))

    noise_curr = gp_model.hyps[-1]
    like_curr = gp_model.like
    err_curr = np.mean(np.abs(pred_vec - truth_vec))

    print(noise_curr)
    print(like_curr)
    print(err_curr)

    noise_pars = np.append(noise_pars, noise_curr)
    likes = np.append(likes, like_curr)
    test_errs = np.append(test_errs, err_curr)

# record likelihood, prediction vector, and hyperparameters
test_err_file = 'test_errs'
like_file = 'like'
noise_file = 'noise'

np.save(test_err_file, test_errs)
np.save(like_file, likes)
np.save(noise_file, noise_pars)
