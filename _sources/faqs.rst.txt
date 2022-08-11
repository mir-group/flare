Frequently Asked Questions
==========================

This page is designed to help users troubleshoot commonly encountered questions when performing tasks in the FLARE framework.


Installation and Packages
-------------------------

1. What do I do if I encounter an mkl error following installation?
        Verify the version of numpy that is installed. Reverting to version 1.18 fixes this error.


Gaussian Processes
------------------

1. How do Gaussian Processes work?
        Gaussian Processes enjoy a long history of study and there are many excellent resources out there we can recommend.
        One such resource (that some of the authors consult quite frequently!) is the textbook
        `Gaussian Processes for Machine Learning, by Rasmussen and Williams <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ 
	, with Chapter 2 in particular being a great help.

2. How should I choose my cutoff(s)?
        * The right cutoff depends on the system you're studying: ionic systems often do better with a larger cutoff, 
        while dense systems like diamond can employ smaller cutoffs. 

        * We recommend that you refer to the radial distribution function of your atomic system first, then try a range of cutoff 
        values and examine the model error, optimized hyperparameters, and model likelihood as a function of the cutoff(s).

        * Keep in mind that the model cutoff is intimately coupled with the radial and angular bases of the model. So, we recommend that 
        model cutoff(s) be tested with varying n_max and l_max.

        * For multi-component systems, a cutoff_matrix is required with explicit cutoffs for each inter-species interaction (e.g., 
        [[1-1, 1-2],[2-1,2-2]] for species 1 and 2), otherwise the matrix will populate with the values of the maximum cutoff listed in the input file.


OTF (On-the-fly) Active-Learning
-------------------------

1. What is a good strategy for hyperparameter optimization?
        Hyperparameter optimization is important for obtaining a good model and is an important step in the case of a bad choice of priors. 
        However, the optimization will get slower when more training data are collected, so some tricks may be needed to provide a good model while minimizing training time.
        There are a few parameters to notice:

        * ``maxiter`` : maximal number of iterations, usually set to ~20 to prevent training for too long. However, if hyperparameter training is unstable,
        raising this number can help if the model is not converged within a smaller number of iterations.

        * ``train_hyps`` : range of DFT calls whererin the hyperparameters will be optimized. We recommend setting the initial value to 10 (i.e., 10th DFT call),
        since we have observed more stable training, as opposed to training from the initial DFT call.

        * ``std_tolerance_factor`` : DFT will be called when the predicted uncertainty is above this threshold, 
        which is defined relative to the mean uncertainty in the system. The default value is 1. In general, we recommend that this value be set relative to the number
        of species in the system (e.g., -0.01 for 1 species, -0.05 for 2, -0.1 for 3, etc.). If more DFT calls are desired, you can set it to a lower value.

        * ``update_threshold`` : atoms will only be added to the sparse set of the Gaussian Process when their uncertainty surpasses this threshold. We have found that this 
        value provides a decent number of sparse environment `additions` when set to be 0.1*std_tolerance_factor. This ensures that several atoms are added to the sparse-set of the
        Gaussian Process for every DFT call. If this value is set to be closer to the std_tolerance_factor, it may be the case where only 1 atomic environment is added for each DFT call,
        which is inefficient depending on the DFT complexity.
        
2. How (why) should a small perturbation be included for the initial structure?
        If you are starting from a perfect lattice, we recommend adding small random perturbations to the atomic positions, 
        such that the symmetry of the crystal lattice is broken. This is accomplished using the `jitter` flag in the `yaml` script, in the units of angstrom.
        The reason is that the perfect lattice is highly symmetric, thus usually the force on each atom is zero, and the local 
        environments all look the same. Adding these highly similar environments with close-to-zero forces might raise numerical
        stability issues for GP.

3. Why is the temperature of the simulation unreasonably high?
        This is the signal of a high-energy configuration being used to start active-learning. Try relaxing the structure before initializing the active-learning trajectory so that your 
        initial structure has atoms in local energy minima. High energy initial structures can yield high forces, leading to instability in the temperature and velocities of the atoms.

        
4. How do I know that my active-learning trajectory is "good"?
        It is important to do some analysis of your active-learning trajectories both while they are running and once they are completed. We recommend that you keep an eye on the system parameters,
        e.g. temperature, pressure, or the radial distribution function. In addition to these system specific markers, we also recommend keeping an eye on the hyperparameters, and making sure that they 
        make sense numerically. 

5. When should I stop my active-learning trajectory?
        Active-learning can be ceased when the number of DFT calls becomes sparse as a function of timestep. The MAE values for energy, forces, and stresses can also indicate when a model has approached a given
        threshold in accuracy. If the number of DFT calls remains low throughout the entire trajectory, try altering the conditions under which the system performs MD (e.g., temperature or pressure) or decrease
        the `std_tolerance_factor` so that more DFT calls will be made.


Offline-Learning 
----

1. Why is my offline training selecting so few sparse environments?
        We have found that it is helpful to reduce the `std_tolerance_factor` below that of what is typically used for active-learning when training a final model with offline learning.
        This is fine, since all of the sparse environments being selected are from DFT calculated frames. It is also helpful to track the likelihood and hyperparameters when reducing this value
        in order to select an appropriate model.

2. How do I know that my offline-trained model is "good"?
        Several markers can be used to evaluate the success of your offline training. Most immediate is the evaluation of errors as assessed throughout training on the DFT frames being used. Also immediately available
        are the hyperparameters, which are based in physical units and should make sense numerically (energy, force, and stress noises relative to the the actual energy, force, and stress lables). The user can also
        generate more in-depth analyses, e.g., parity plots of energies, forces, and stresses. 



Production MD Simulations using a FLARE ML-FF
----

1. Which MD engines is FLARE compatible?
        We commonly employ our trained FLARE models in LAMMPs and the ASE md engines.

2. How do I know that my model is performing well?
        Without diving into system-specific benchmarks that can be done, we recommend using the uncertainty quantification capabilities of FLARE to determine whether your MD simulation is operating within the domains of the 
        training set. Example scripts for the quantification of uncertianty can be found elsewhere in this repository.

3. Why is my simulation misbehaving?
        Several parameters can influence the success of the MD simulations that are run after building your FLARE model. It is important to first check that the species match the order that is present in the
        lammps coefficient file, and that their masses are assigned appropriately. 
        - If non-physical environments appear in your simulation (either by visual inspection or via uncertainty analysis), several tricks can be implemented to fix this.
                (1) try reducing the timestep. An aggressive timestep can lead to errors in integration and prompt unphysical environments to appear. 
                (2) toggle the thermostat damping factor (specific to the MD engine being used).
                (3) make sure that the initial structure is reasonable and not unreasonably high in energy or does not have high forces. (related to next point)

        - If the temperature of the simulation is unreasonably high upon initialization:
                (1) try relaxing the structure using built-in methods (e.g., conjugate gradient descent in LAMMPS) so that your initial structure has atoms in local energy minima. High energy initial structures
                can yield high forces, leading to temperature increasing drastically.
