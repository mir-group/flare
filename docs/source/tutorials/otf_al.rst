On-the-fly aluminum potential
=============================

Here we give an example of running OTF (on-the-fly) training with QE (Quantum Espresso) and NVE ensemble. 
We use our unit test file as illustration (``test_OTF_qe.py``)

Step 1: Set up a GP Model 
-------------------------

Let's start up with the GP model with three-body kernel function. 
(See :doc:`kernels.py <../flare/kernels>` (single component)
or :doc:`mc_simple.py <../flare/mc_simple>` (multi-component) for more options.)

.. code-block:: python
    :linenos:

    # make gp model
    kernel = en.three_body
    kernel_grad = en.three_body_grad
    hyps = np.array([0.1, 1, 0.01])
    hyp_labels = ['Signal Std', 'Length Scale', 'Noise Std']
    cutoffs = np.array([3.9, 3.9])
    energy_force_kernel = en.three_body_force_en

    gp = \
        GaussianProcess(kernel=kernel,
                        kernel_grad=kernel_grad,
                        hyps=hyps,
                        cutoffs=cutoffs,
                        hyp_labels=hyp_labels,
                        energy_force_kernel=energy_force_kernel,
                        maxiter=50)


**Some Explanation about the parameters:**

* ``kernel``: set to be the kernel function  

    * import from :doc:`kernels.py <../flare/kernels>` (single-component system) 
      or :doc:`mc_simple.py <../flare/mc_simple>` (multi-component system). 
    * Currently we have the choices of two-body, three-body and two-plus-three-body kernel functions.
    * Two-plus-three-body kernel function is simply the summation of two-body and three-body kernels,
      and is tested to have best performance.

* ``kernel_grad``: set to be the gradient of kernel function used for hyperparameter training. 

* ``hyps``: the array of hyperparameters, whose names are shown in ``hyp_labels``.

    * For two-body kernel function, an array of length 3 is needed, ``hyps=[sigma_2, ls_2, sigma_n]``;
    * For three-body, ``hyps=[sigma_3, ls_3, sigma_n]``;
    * For two-plus-three-body, ``hyps=[sigma_2, ls_2, sigma_3, ls_3, sigma_n]``.

* ``cutoffs``: consists of two values. The 1st is the cutoff of two-body and the 2nd is for three-body kernel. 
  Usually we will set a larger one for two-body.

* ``energy_force_kernel``: set to calculate local energy for each atom based on the integral of forces. 

* ``maxiter``: set to constrain the number of steps in training hyperparameters. 


**Note:**

1. See :doc:`GaussianProcess <../flare/gp>` for complete description of arguments of ``GaussianProcess`` class.

2. In any case, you need to set up ``kernel``. 

    a. If you want to train the hyperparameters, you need to set up ``kernel_grad``;
    b. If you want the output of local energy, you need to set up ``energy_force_kernel`` and ``energy_kernel``, 
       otherwise you don't need them.


Step 2: Set up DFT Calculator
-----------------------------

The next step is to set up DFT calculator, here we use QE (quantum espresso). 
Suppose we've prepared a QE input file in current directory ``./pwscf.in``, 
and have set the environment variable ``PWSCF_COMMAND`` to the location of our QE's executable ``pw.x``. 
Then we specify the input file and executable by ``qe_input`` and ``dft_loc``.

.. code-block:: python
    :linenos:

    # set up DFT calculator
    qe_input = './pwscf.in' # quantum espresso input file
    dft_loc = os.environ.get('PWSCF_COMMAND') 
 

Step 3: Set up OTF MD Training Engine
--------------------------------------------------
Then we can set up our On-The-Fly (OTF) MD engine for training and simulation. 

.. code-block:: python
    :linenos:

    # set up OTF parameters
    dt = 0.001                  # timestep (ps)
    number_of_steps = 100       # number of steps
    std_tolerance_factor = 1   
    max_atoms_added = 2
    freeze_hyps = 3

    otf = OTF(qe_input, dt, number_of_steps, gp, dft_loc,
              std_tolerance_factor, init_atoms=[0],
              calculate_energy=True, output_name='al_otf_qe',
              freeze_hyps=freeze_hyps, skip=5,
              max_atoms_added=max_atoms_added)


**Some Explanation about the parameters:**

* ``dt``: the time step in unit of *ps*
* ``number_of_steps``: the number of steps that the MD is run
* ``std_tolerance_factor``: the uncertainty threshold = std_tolerance_factor x hyps[-1]. 
  In OTF training, when GP predicts uncertainty above the uncertainty threshold, it will call DFT
* ``max_atoms_added``: constrain the number of atoms added to the training set after each DFT call
* ``freeze_hyps``: stop training hyperparameters and fix them from the ``freeze_hyps`` th step. 
  Usually set to a small number, because for large dataset the training will take long.
* ``init_atoms``: list of atoms to be added in the first DFT call. 
  Because there's no uncertainty predicted in the initial DFT call, 
  so there's no selection rule to pick up "maximully uncertain" atoms into the training set, 
  we have to specify which atoms to pick up by this variable.
* ``calculate_energy``: if ``True``, the local energy on each atom will be calculated
* ``output_name``: the name of the logfile
* ``skip``: record/dump the information every ``skip`` steps.


Step 4: Launch the OTF Training
-------------------------------

Finally, let's run it!

.. code-block:: python
    :linenos:

    # run OTF MD
    otf.run()


After OTF training is finished, we can check log file ``al_otf_qe.out`` for all the information dumped. 
This output file can be parsed using our ``otf_parser.py`` module, which we will give an introduction later.
