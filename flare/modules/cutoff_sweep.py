import numpy as np
import sys
import gp, env, struc, kernels, analyze_gp, qe_parsers
from scipy.optimize import minimize
import time
import datetime


def sweep(txt_name, data_file, cell, training_snaps, cutoffs, kernel,
          bodies, nos):

    # set up text file
    txt = update_init()
    write_file(txt_name, txt)

    # define md_trajectory object
    md_trajectory = analyze_gp.MDAnalysis(data_file, cell)

    # set up training run
    hyps = np.array([1, 1, 1])

    for cutoff in cutoffs:
        gp_test = \
            analyze_gp.get_gp_from_snaps(md_trajectory, training_snaps, cutoff,
                                         kernel, bodies, nos=nos)

        gp_test.algo = 'BFGS'
        gp_test.hyps = hyps

        # train gp model
        time0 = time.time()
        gp_test.train(monitor=True)
        time1 = time.time()
        training_time = time1 - time0

        likelihood = -gp_test.get_likelihood(gp_test.hyps,
                                             gp_test.training_data,
                                             gp_test.training_labels_np,
                                             gp_test.kernel, gp_test.bodies)

        hyps = gp_test.hyps

        txt += """\n
    cutoff: {}
    optimized hyperparameters:
    """.format(cutoff)
        txt += str(hyps)
        txt += """
    likelihood: %.5f
    training time: %.2f s""" % (likelihood, training_time)

        write_file(txt_name, txt)


def sweep_and_test(txt_name, data_file, cell, training_snaps, cutoffs, kernel,
                   bodies, nos, test_snaps):
        # set up text file
    txt = update_init()
    write_file(txt_name, txt)

    # define md_trajectory object
    md_trajectory = analyze_gp.MDAnalysis(data_file, cell)

    # set up training run
    hyps = np.array([1, 1, 1])

    for cutoff in cutoffs:
        gp_test = \
            analyze_gp.get_gp_from_snaps(md_trajectory, training_snaps, cutoff,
                                         kernel, bodies, nos=nos)

        gp_test.algo = 'BFGS'
        gp_test.hyps = hyps

        # train gp model
        time0 = time.time()
        gp_test.train(monitor=True)
        time1 = time.time()
        training_time = time1 - time0

        likelihood = -gp_test.get_likelihood(gp_test.hyps,
                                             gp_test.training_data,
                                             gp_test.training_labels_np,
                                             gp_test.kernel, gp_test.bodies)

        hyps = gp_test.hyps

        txt += """\n
    cutoff: {}
    optimized hyperparameters:
    """.format(cutoff)
        txt += str(hyps)
        txt += """
    likelihood: %.5f
    training time: %.2f s""" % (likelihood, training_time)

        write_file(txt_name, txt)

        # test model
        all_predictions, all_variances, all_forces = \
            analyze_gp.predict_forces_on_test_set(gp_test, md_trajectory,
                                                  test_snaps, cutoff)

        training_set_size = len(training_snaps) * 32 * 3
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
\n""" % (training_set_size, avg_force, max_force, mae, max_err, avg_std,
         max_std)

        write_file(txt_name, txt)


def write_file(fname, text):
    with open(fname, 'w') as fin:
        fin.write(text)


def update_init():
    init_text = """Two body cutoff test.
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
