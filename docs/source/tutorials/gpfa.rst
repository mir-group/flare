Training a Gaussian Process from an AIMD Run 
===============================================
Steven Torrisi (torrisi@g.harvard.edu), December 2019

In this tutorial, we'll demonstrate how a previously existing Ab-Initio 
Molecular  Dynamics (AIMD) trajectory can be used to train a Gaussian Process model.

We can use a very short trajectory for a very simple molecule which is already 
included in the test files in order to demonstrate how to set up and run the code.
The trajectory this tutorial focuses on  involves a few frames of the 
molecule Methanol vibrating about it's equilibrium configuration, ran in VASP. 



 Step 1: Setting up a Gaussian Process Object
--------------------------------------------------
Our goal is to train a GP, which first must be instantiated with a set of parameters.

For the sake of this example, which is a molecule, we will use a two-plus-three body kernel. 
We must provide the kernel and the kernel gradient as callables to the GP. 
Our initial guesses for the hyperparameters are not important. 
The hyperparameter labels are included below for later output.
The system contains a small number of atoms, so we choose a relatively 
smaller 2-body cutoff (7 A) and a relatively large 3-body cutoff (7 A), both of which will completely contain the molecule.

At the header of a file, include the following imports:
.. codeblock:: python
	from flare.gp import GaussianProcess
	from flare.mc_simple import two_plus_three_body_mc, two_plus_three_body_mc_grad

We will then set up the ``GaussianProcess`` object.

* The ``GaussianProcess`` object class contains the methods which, from an 
	``AtomicEnvironment`` object, predict the corresponding forces and 
	uncertainties by comparing the atomic environment to each environment in the
	training set. The kernel we will use has 5 hyperparameters and requires two cutoffs. 
* The first four hyperparameters correspond to the signal variance and length 
	scale which parameterize the two- and three-body comparison 
	functions. These hyperparameters will be optimized later once data has 
	been fed into the ``GaussianProcess`` via likelihood maximization. The 
	fifth and final hyperparameter is the noise variance. We provide simple 
	initial guesses for each hyperparameter.
* The two cutoff values correspond to the functions which set up 
	the two- and three-body Atomic Environments. Since Methanol is a small 
	molecule, 7 Angstrom each will be sufficent.
* The kernels which facilitate these comparisons must be imported as Python  ``callable``s. 
* Here, we will use the ``two_plus_three_body_mc`` kernel, which 
	uses two-body and three-body comparisons. ``mc`` means multi-component, 
	indicating that it can handle multiple atomic species being present.
* We must also import the gradient of the kernel, which is
	``two_plus_three_body_mc_grad``.
 

.. codeblock:: python
	gp = GaussianProcess(kernel=two_plus_three_body_mc, kernel_grad=two_plus_three_body_mc_grad,
			hyps=[0.01, 0.01, 0.01, 0.01, 0.01],
			cutoffs = (7,7),
			hyp_labels=['Two-Body Signal Variance','Two-Body Length Scale','Three-Body Signal Variance',
					'Three-Body Length Scale', 'Noise Variance']
			)
