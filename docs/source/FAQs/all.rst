Frequently Asked Questions
==========================

Installation and Packages
-------------------------
1. What numba version will I need?

2. Can I accelerate my numpy installation using parallelization?

Gaussian Processes and OTF
--------------------------


1. I'm confused about how Gaussian Processes work.
        Gaussian Processes enjoy a long history of study and there are many excellent resources out there we can recommend.
        One such resource (that some of the authors consult quite frequently!) is the textbook
        `Gaussian Processes for Machine Learning, by Rasmussen and Williams <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ 
	, with Chapter 2 in particular being a great help.



2. How should I choose my cutoffs?

3. What is a good strategy for hyperparameter optimization?	

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
