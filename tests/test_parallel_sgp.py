import os, sys, shutil, time
import numpy as np
import pytest

from ase.io import read, write
from flare.atoms import FLARE_Atoms

from flare.bffs.sgp.calculator import SGP_Calculator

from .get_sgp import get_random_atoms, species_map, get_empty_parsgp, get_empty_sgp, get_training_data

# we set the same seed for different ranks, 
# so no need to broadcast the structures

np.random.seed(10)

def test_update_db():
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    # build a non-empty parallel sgp
    sgp = get_empty_parsgp()
    training_strucs, training_sparse_indices = get_training_data()
    sgp.build(training_strucs, training_sparse_indices, update=False)

    # add a new structure
    train_structure = get_random_atoms(a=2.0, sc_size=2, numbers=list(species_map.keys()))
    sgp.update_db(train_structure, custom_range=[2, 3], mode="uncertain")

    u_size = 0
    for k in range(len(sgp.descriptor_calculators)):
        u_size_kern = 0
        for inds in sgp.training_sparse_indices[k]:
            u_size_kern += len(inds)
        u_size += u_size_kern

    assert sgp.sparse_gp.Kuu.shape[0] == u_size
    assert sgp.sparse_gp.alpha.shape[0] == u_size

    sgp.sparse_gp.finalize_MPI = False

def test_train():
    """Check that the hyperparameters and likelihood are updated when the
    train method is called."""

    # TODO: add sparse_gp and compare the results

    #from flare_pp.sparse_gp import compute_negative_likelihood_grad_stable
    #new_hyps = np.array(sgp.hyps) + 1
    ##
    ##tic = time.time()
    #compute_negative_likelihood_grad_stable(new_hyps, sgp.sparse_gp, precomputed=False)
    ##toc = time.time()
    #print("compute_negative_likelihood_grad_stable TIME:", toc - tic)

    # build a non-empty parallel sgp
    sgp = get_empty_parsgp()
    training_strucs, training_sparse_indices = get_training_data()
    sgp.build(training_strucs, training_sparse_indices, update=False)

    hyps_init = tuple(sgp.hyps)
    sgp.train()
    hyps_post = tuple(sgp.hyps)

    #assert hyps_init != hyps_post
    assert sgp.likelihood != 0.0

    sgp.sparse_gp.finalize_MPI = False

def test_predict():
    # build a non-empty parallel sgp
    training_strucs, training_sparse_indices = get_training_data()
    sgp = get_empty_parsgp()
    sgp.build(training_strucs, training_sparse_indices, update=False)

    # build serial sgp with the same training data set
    sgp_serial = get_empty_sgp()
    for t in range(len(training_strucs)):
        sgp_serial.update_db(
            training_strucs[t],
            training_strucs[t].forces,
            custom_range=[training_sparse_indices[k][t] for k in range(len(sgp_serial.descriptor_calculators))],
            energy=training_strucs[t].potential_energy,
            stress=training_strucs[t].stress,
            mode="specific",
        )
    sgp_serial.sparse_gp.update_matrices_QR()

    # generate testing data
    n_frames = 5
    test_strucs = []
    for n in range(n_frames): 
        atoms = get_random_atoms(a=2.0, sc_size=2, numbers=list(species_map.keys()))
        test_strucs.append(atoms)

    # predict on testing data
    sgp.predict_on_structures(test_strucs)
    for n in range(n_frames):
        par_energy = test_strucs[n].get_potential_energy()
        par_forces = test_strucs[n].get_forces()
        par_stress = test_strucs[n].get_stress()

        test_strucs[n].calc = SGP_Calculator(sgp_serial)

        ser_energy = test_strucs[n].get_potential_energy()
        ser_forces = test_strucs[n].get_forces()
        ser_stress = test_strucs[n].get_stress()

        np.allclose(par_energy, ser_energy)
        np.allclose(par_forces, ser_forces)
        np.allclose(par_stress, ser_stress)

    sgp.sparse_gp.finalize_MPI = True
