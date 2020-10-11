1. Merge multiple GP models
===========================

There are situations where we train several GP models separately, and
need to merge them into one model. Here is the instruction.

Suppose we have two GP models from files ``gp1`` and ``gp2``, we need to

Step 1: Concatenate the training data of the two models together

.. code:: ipython3

    import numpy as np
    from flare.gp import GaussianProcess
    
    # load GP models from file
    gp1 = GaussianProcess.from_file("gp_model_1.json")
    gp2 = GaussianProcess.from_file("gp_model_2.json")
    
    # append the training data set of gp2 to that of gp1
    gp_tot = gp1
    gp_tot.training_data += gp2.training_data
    gp_tot.training_labels += gp2.training_labels
    gp_tot.training_labels_np = np.array(gp_tot.training_labels)
    
    # Optional: if you've also included total energy label into the training set
    gp_tot.training_structures += gp2.training_structures
    gp_tot.energy_labels += gp2.energy_labels
    gp_tot.energy_labels_np = np.array(gp_tot.energy_labels_np)
    
    # sync data
    gp_tot.all_labels = np.concatenate((gp_tot.training_labels_np, gp_tot.energy_labels_np))
    gp_tot.sync_data()

Step 2: Train the hyperparameters of the merged GP model

.. code:: ipython3

    gp_tot.train()

Step 2*: If you don’t want to train the hyperparameters, you still need
to call ``set_L_alpha()`` once to compute the new kernel matrix. If you
called ``train()``, then you don’t need this step because the kernel
matrix is computed during training.

.. code:: ipython3

    gp_tot.set_L_alpha()

Step 3: Write model to file

.. code:: ipython3

    gp_tot.write_model("gp_model_tot.json")
