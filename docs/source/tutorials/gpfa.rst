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
