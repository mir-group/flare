Frequently Asked Questions
==========================

Installation and Packages
-------------------------
1. What numba version will I need?
        >= 0.50.0
        The latest version is always recommended.

**Note:** If you get errors with `numba` or get `C/C++`-type errors, 
very possibly it's the problem of the `numba` version.

2. Can I accelerate my calculation using parallelization?
        See the section in the `Installation <https://flare.readthedocs.io/en/latest/install.html#acceleration-with-multiprocessing-and-mkl>`_ section.

Gaussian Processes
------------------


1. I'm confused about how Gaussian Processes work.
        Gaussian Processes enjoy a long history of study and there are many excellent resources out there we can recommend.
        One such resource (that some of the authors consult quite frequently!) is the textbook
        `Gaussian Processes for Machine Learning, by Rasmussen and Williams <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ 
	, with Chapter 2 in particular being a great help.


2. How should I choose my cutoffs?
        * The right cutoff depends on the system you're studying: ionic systems often do better with a larger 2-body cutoff, 
        while dense systems like diamond require smaller cutoffs. 

        * We recommend you to refer to the radial distribution function of your atomic system first, then try a range of cutoff 
        values and examine the model error, optimized noise parameter, and model likelihood as a function of the cutoff.

        * In the current implementation, the 3-body cutoff needs to be smaller than 2-body cutoff.

3. What is a good strategy for hyperparameter optimization?	
        The hyperparameter optimization is important for obtaining a good model. 
        However, the optimization of hyperparameters will get slower when more training data are collected.
        There are a few parameters to notice:
        
        In `GaussianProcess`,

        * ``maxiter``: maximal number of iterations, usually set to ~10 to prevent training for too long.

        * ``parallel``: if `True`, then parallelization is used in optimization. 
          The serial version could be very slow.

        * ``output`` (in `train` function): set up an output file for monitoring optimization iterations.

        * ``grad_tol``, ``x_tol``, ``line_steps`` (in `train` function): can be changed for iterations.

        There are a few tips in the OTF training, see below.
            


OTF (On-the-fly) Training
-------------------------

1. What is a good strategy for hyperparameter optimization?
        * ``freeze_hyps`` : the hyperparameter will only be optimized for `freeze_hyps` times. 
          Can be set to a small number if optimization is too expensive with a large data set.

        * ``std_tolerance_factor`` : the DFT will be called when the predicted uncertainty is above the threshold, 
          which is defined as `std_tolerance_factor * gp_hyps_noise`. The default value is 1. In general, you 
          can set it to O(0.1)-O(1). If more DFT calls are desired, you can set it to a lower value.

2. How to set initial temperatures and rescale temperatures?
        * For initial temperature, if you are using NVE ensemble, and starting from a perfect crystal lattice, 
          please set the initial temperature to be twice the value you want the system to be in equilibrium. E.g.,
          if you want the system to equilibrate at 1000K, set the initial temperature to 2000K starting from a 
          perfect lattice. To do this,
                * if you are using our OTF trainer without ASE, you can set the ``prev_positions`` of the initial 
                  structure to be the perfect lattice shifted by one step with the velocity corresponding to the 
                  initial temperature.
                * if you are using OTF with ASE + VelocityVerlet, you can set the initial velocity as shown in our
                  tutorial

        * Similarly, if you want to rescale the system from 300K to 500K at step 1000, you should set the resaling 
          temperature to be higher than 500K, e.g.
          
          .. code-block:: python

                rescale_temp = [800]
                rescale_step = [1000]

          The reason is that we only rescale the velocity of the atoms at the steps specified in ``rescale_step``, 
          at those steps not in the list, we will let the system evolve by itself. Thus, after step 1000, the system's
          temperature will gradually equilibrate at a lower temperature.
        
3. Include small perturbation for initial structure
        If you are starting from a perfect lattice, we recommend adding small random perturbations to the atomic positions, 
        such that the symmetry of the crystal lattice is broken. E.g.

        .. code-block:: python

                positions = positions + 0.01 * (2 * np.random.rand(len(positions), 3) - 1)

        The reason is that the perfect lattice is highly symmetric, thus usually the force on each atom is zero, and the local 
        environments all look the same. Adding these highly similar environments with close-to-zero forces might raise numerical
        stability issue for GP.

        
GPFA 
----

1. My models are adding too many atoms from each frame, causing a serious slowdown without much gain in model accuracy.
	In order to 'govern' the rate at which the model adds atoms, we suggest using the ``pre_train_atoms_per_element`` and
	``train_atoms_per_element`` arguments, which can limit the number of atoms added from each seed frame and training frame respectively.
	You can pass in a dictionary like ``{'H':1, 'Cu':2}`` to limit the number of H atoms to 1 and Cu atoms to 2 from any given frame.
	You can also use ``max_atoms_per_frame`` for the same functionality.
2. The uncertainty seems low on my force predictions, but the true errors in the forces are high.
	This could be happening for a few reasons. One reason could be that your hyperparameters aren't at an optimum (check that the gradient of
	the likelihood with respect to the hyperparameters is small). Another is that your model, such as 2-body or 2+3 body, may not be of sufficient 
	complexity to handle the system (in other words, many-body effects could be important).

MGP
---
1. How does the grid number affect my mapping?
        * The lower cutoff is better set to be a bit smaller than the minimal interatomic distance.
        * The upper cutoff should be consistent with GP's cutoff. 
        * For three-body, the grid is 3-D, with lower cutoffs `[a, a, a]` and upper cutoffs `[b, b, b]`.
        * You can try different grid numbers and compare the force prediction of MGP and GP 
          on the same testing structure. Choose the grid number of satisfying efficiency and accuracy.
          A reference is `grid_num=64` should be safe for `a=2.5`, `b=5`.
