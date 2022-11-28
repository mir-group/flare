Fake MD
=======

The fake MD and fake DFT modules provide interfaces to OTF, such that we can do 
offline training, i.e. use a given ab-initio MD trajectory to train a GP 
potential with active learning. 

In the OTF training, instead of running a real MD step to get to the next frame,
fake MD feeds the next frame from the given trajectory. And instead of using
real DFT for ground truth calculation, fake DFT reads the forces from the given 
trajectory as ground truth.

.. automodule:: flare.md.fake
    :members:
