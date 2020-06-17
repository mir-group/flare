Prepare your data
=================

If you have collected data for training, including atomic positions, chemical 
species, cell etc., you need to convert it into a list of ``Structure`` objects. 
Below we provide a few examples.


VASP data
---------

If you have AIMD data from VASP, you can follow 
`the step 2 of this instruction <https://flare.readthedocs.io/en/latest/tutorials/gpfa.html>`_
to generate ``Structure`` with the ``vasprun.xml`` file. 


Data from Quantum Espresso, LAMMPS, etc.
----------------------------------------

If you have collected data from any 
`calculator that ASE supports <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_,
or have dumped data file of `format that ASE supports <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_,
you can convert your data into ASE ``Atoms``, then from ``Atoms`` to 
``Structure`` via ``Structure.from_ase_atoms``.

For example, if you have collected data from QE, and obtained the QE output file ``.pwo``, 
you can parse it with ASE, and convert ASE ``Atoms`` into ``Structure``.


.. code-block:: python
 
    from ase.io import read
    from flare.struc import Structure

    frames = read('data.pwo', index=':', format='espresso-out') # read the whole traj
    trajectory = []
    for atoms in frames:
        trajectory.append(Structure.from_ase_atoms(atoms))


If the data is from the LAMMPS dump file, use

.. code-block:: python
    
    # if it's text file
    frames = read('data.dump', index=':', format='lammps-dump-text')

    # if it's binary file
    frames = read('data.dump', index=':', format='lammps-dump-binary')


Then the ``trajectory`` can be used to
`train GP from AIMD data <https://flare.readthedocs.io/en/latest/tutorials/gpfa.html>`_.


Try building GP from data
-------------------------

To have a more complete and better monitored training process, please use our 
`GPFA module <https://flare.readthedocs.io/en/latest/tutorials/gpfa.html>`_. 

Here we are not going to use this module, but only provide a simple example on 
how the GP is constructed from the data.

.. code-block:: python

    from flare.gp import GaussianProcess
    from flare.utils.parameter_helper import ParameterHelper

    # set up hyperparameters, cutoffs
    kernels = ['twobody', 'threebody']
    parameters = {'cutoff_twobody': 4.0, 'cutoff_threebody': 3.0}
    pm = ParameterHelper(kernels=kernels, 
                         random=True,
                         parameters=parameters)
    hm = pm.as_dict()
    hyps = hm['hyps']
    cutoffs = hm['cutoffs']
    hl = hm['hyp_labels']

    kernel_type = 'mc' # multi-component. use 'sc' for single component system

    # build up GP model
    gp_model = \
        GaussianProcess(kernels=kernels,
                        component=kernel_type,
                        hyps=hyps,
                        hyp_labels=hl,
                        cutoffs=cutoffs, 
                        hyps_mask=hm,
                        parallel=False, 
                        n_cpus=1)

    # feed training data into GP
    # use the "trajectory" as from above, a list of Structure objects
    for train_struc in trajectory: 
        gp_model.update_db(train_struc, forces)
    gp_model.check_L_alpha() # build kernel matrix from training data

    # make a prediction with gp, test on a training data
    test_env = gp_model.training_data[0]
    gp_pred = gp_model.predict(test_env, 1) # obtain the x-component 
                                            # (force_x, var_x)
                                            # x: 1, y: 2, z: 3
    print(gp_pred)
