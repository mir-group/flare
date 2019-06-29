import numpy as np
import sys
from flare import gp, struc
from flare.modules import analyze_gp
import time
import datetime


def data_sweep(data_file, cell, training_snaps, cutoffs, kernel, kernel_grad,
               hyps, test_snaps, txt_name):
    """Uses coordinates and forces from a Quantum Espresso AIMD output file to
    test the performance of a GP model as a function of the training set size.
    """

    md_trajectory = analyze_gp.MDAnalysis(data_file, cell)

    # set up text file
    txt = update_init()
    write_file(txt_name, txt)

    # initialize gp model
    gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps,
                                  cutoffs, opt_algorithm='BFGS')

    for n, snap in enumerate(training_snaps):
        # update gp
        time0 = time.time()
        structure = md_trajectory.get_structure_from_snap(snap)
        forces = md_trajectory.get_forces_from_snap(snap)
        gp_model.update_db(structure, forces)
        gp_model.set_L_alpha()
        time1 = time.time()

        # predict on test set
        all_predictions, all_variances, all_forces = \
            analyze_gp.predict_forces_on_test_set(gp_model, md_trajectory,
                                                  test_snaps, cutoffs)
        time2 = time.time()

        update_time = time1 - time0
        prediction_time = time2 - time1

        # record predictions for parity plots
        if n == 0:
            np.save('aimd_forces', all_forces)
        pred_name = 'preds_' + str(snap)
        vars_name = 'vars_' + str(snap)
        np.save(pred_name, all_predictions)
        np.save(vars_name, all_variances)

        # compute and report error
        training_set_size = len(gp_model.training_labels_np)
        avg_force = np.mean(np.abs(all_forces))
        max_force = np.max(np.abs(all_forces))
        mae = np.mean(np.abs(all_predictions - all_forces))
        max_err = np.max(np.abs(all_predictions - all_forces))
        avg_std = np.mean(np.sqrt(all_variances))
        max_std = np.max(np.sqrt(all_variances))

        txt += """\n
training_set_size = %i
average force = %.4f
max force = %.4f
mean absolute error = %.4f
max error = %.4f
average std = %.4f
max std = %.4f
update time (s) = %.4f
prediction time (s) = %.4f
\n""" % (training_set_size, avg_force, max_force, mae, max_err, avg_std,
         max_std, update_time, prediction_time)

        write_file(txt_name, txt)


def data_sweep_update(data_file, cell, training_snaps, cutoffs, kernel,
                      kernel_grad, hyps, test_snaps, txt_name):
    """Uses coordinates and forces from a Quantum Espresso AIMD output file to
    test the performance of a GP model as a function of the training set size.
    """

    init_time = time.time()
    md_trajectory = analyze_gp.MDAnalysis(data_file, cell)

    # set up text file
    txt = update_init()
    write_file(txt_name, txt)

    # initialize gp model
    gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps,
                                  cutoffs, opt_algorithm='BFGS')

    for n, snap in enumerate(training_snaps):
        # update gp
        structure = md_trajectory.get_structure_from_snap(snap)
        forces = md_trajectory.get_forces_from_snap(snap)
        gp_model.update_db(structure, forces)

        if n == 0:
            gp_model.set_L_alpha()
        else:
            gp_model.update_L_alpha_v1()

        # predict on test set
        all_predictions, all_variances, all_forces = \
            analyze_gp.predict_forces_on_test_set(gp_model, md_trajectory,
                                                  test_snaps, cutoffs)

        # compute and report error
        training_set_size = len(gp_model.training_labels_np)
        avg_force = np.mean(np.abs(all_forces))
        max_force = np.max(np.abs(all_forces))
        mae = np.mean(np.abs(all_predictions - all_forces))
        max_err = np.max(np.abs(all_predictions - all_forces))
        avg_std = np.mean(np.sqrt(all_variances))
        max_std = np.max(np.sqrt(all_variances))
        time_curr = time.time()
        time_from_start = time_curr - init_time

        txt += """\n
training_set_size = %i
average force = %.4f
max force = %.4f
mean absolute error = %.4f
max error = %.4f
average std = %.4f
max std = %.4f
time from start (s) = %.4f
\n""" % (training_set_size, avg_force, max_force, mae, max_err, avg_std,
         max_std, time_from_start)

        write_file(txt_name, txt)


def write_file(fname, text):
    with open(fname, 'w') as fin:
        fin.write(text)


def update_init():
    init_text = """Data sweep.
Date and time: %s.
Author: Jonathan Vandermause.
""" % str(datetime.datetime.now())

    return init_text


def update_fin():
    fin_text = """
-------------------------------------------------------------------------------
   JOB DONE.
-------------------------------------------------------------------------------
"""
    return fin_text
