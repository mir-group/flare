Mapped Gaussian Process
=======================

.. toctree::
   :maxdepth: 2

   splines_methods

Take 2-body as an example. Consider the atomic environment :math:`\rho`, whose force
is to be predicted via GP regression:

.. math::

    E(\rho) = \sum_i k(\rho, \rho_i) \alpha_i

.. automodule:: flare.mgp.mgp
    :members:
