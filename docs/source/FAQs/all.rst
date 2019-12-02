Frequently Asked Questions
==========================

Installation and Packages
-------------------------
1. What numba version will I need?
        0.43.1 or greater.

2. Can I accelerate my calculation using parallelization?
        See the section in the `Installation <https://flare.readthedocs.io/en/latest/install.html#acceleration-with-multiprocessing-and-mkl>`_ section.

Gaussian Processes and OTF
--------------------------


1. I'm confused about how Gaussian Processes work.
        Gaussian Processes enjoy a long history of study and there are many excellent resources out there we can recommend.
        One such resource (that some of the authors consult quite frequently!) is the textbook
        `Gaussian Processes for Machine Learning, by Rasmussen and Williams <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ 
	, with Chapter 2 in particular being a great help.


2. How should I choose my cutoffs?
        The right cutoff depends on the system you're studying: ionic systems often do better with a larger 2-body cutoff, while dense systems like diamond reqeuire smaller 2- and 3-body cutoffs. We recommend you try a range of cutoff values and examine the model error, optimized noise parameter, and model likelihood as a function of the cutoff.

3. What is a good strategy for hyperparameter optimization?	
        Start with a plausible guess (e.g. set the length scale hyperparameters to 1 A and the noise hyperparameter to 0.1 eV/A). If the likelihood gradient is ill-behaved, add more data to the GP and re-train.

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
