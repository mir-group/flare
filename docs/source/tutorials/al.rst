On-the-fly aluminum potential
=============================

Here we give an example of running OTF (on-the-fly) training with QE (Quantum Espresso) and NVE ensemble. We use our unit test file as illustration (`test_OTF_qe.py`)

First, let's set up a GP model with three-body kernel function.

* `kernel_grad`: set to be the gradient of kernel function used for hyperparameter training. 
* `hyps`: the array of hyperparameters, whose names are shown in `hyp_labels`.
* `cutoffs`: consists of two values. The 1st is the cutoff of two-body and the 2nd is for three-body kernel. Usually we will set a larger one for two-body.
* `energy_force_kernel`: set to calculate local energy for each atom based on the integral of forces. 
* `maxiter`: set to constrain the number of steps in training hyperparameters. 

**Note:** If you are trying multi-component system, please use kernel functions in `mc_simple.py` instead of `kernels.py`

.. literalinclude:: ../../../tests/test_OTF_qe.py
   :lines: 82-97

The next step is to set up DFT calculator, here we use QE (quantum espresso). Suppose we've prepared a QE input file in current directory `./pwscf.in`, and have set the environment variable `PWSCF_COMMAND` to the location of our QE's executable `pw.x`. Then we specify the input file and executable by `qe_input` and `dft_loc`.

.. literalinclude:: ../../../tests/test_OTF_qe.py
   :lines: 99-101

Then we can set up our OTF MD engine. 

* `dt`: the time step in unit of *ps*
* `number_of_steps`: the number of steps that the MD is run
* `std_tolerance_factor`: the uncertainty threshold = std_tolerance_factor x hyps[-1]. In OTF training, when GP predicts uncertainty above the uncertainty threshold, it will call DFT
* `max_atoms_added`: constrain the number of atoms added to the training set after each DFT call
* `freeze_hyps`: stop training hyperparameters and fix them from the `freeze_hyps` th step. Usually set to a small number, because for large dataset the training will take long.
* `init_atoms`: list of atoms to be added in the first DFT call. Because there's no uncertainty predicted in the initial DFT call, so there's no selection rule to pick up "maximully uncertain" atoms into the training set, we have to specify which atoms to pick up by this variable.
* `calculate_energy`: if `True`, the local energy on each atom will be calculated
* `output_name`: the name of the logfile
* `skip`: record/dump the information every `skip` steps.

.. literalinclude:: ../../../tests/test_OTF_qe.py
   :lines: 103-114

Finally, let's run it!

.. literalinclude:: ../../../tests/test_OTF_qe.py
   :lines: 116-117

After OTF training is finished, we can check log file `al_otf_qe.out` for all the information dumped. This output file can be parsed using our `otf_parser.py` module, which we will give an introduction later.
