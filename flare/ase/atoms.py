import os
import sys
import inspect
from time import time
from copy import deepcopy

import numpy as np
from ase import Atoms
from flare.utils.learner import get_max_cutoff

class FLARE_Atoms(Atoms):
    '''
    The `FLARE_Atoms` class is a child class of ASE `Atoms`, 
    which has completely the same usage as the primitive ASE `Atoms`, and
    in the meanwhile mimic `Structure` class. It is used in the `OTF` module
    with ASE engine (by `OTF_ASE` module). It enables attributes to be 
    obtained by both the name from ASE `Atoms` and `Structure`.

    The input arguments are the same as ASE `Atoms`.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_positions = np.zeros_like(self.positions)

    @static
    def from_ase_atoms(atoms):
        return FLARE_Atoms(**atoms.__dict__)

    @property
    def nat(self):
        return len(self)

    @property
    def species_labels(self):
        return self.symbols

    @property
    def coded_species(self):
        return self.numbers

    @property
    def forces(self):
        return self.atoms.calc.get_forces()

    @property
    def potential_energy(self):
        return self.atoms.calc.get_potential_energy()

    @property
    def stress(self):
        return self.atoms.calc.get_stress()

    @property
    def stress_stds(self):
        raise NotImplementedError

    @property
    def local_energy_stds(self):
        raise NotImplementedError

    @property
    def stds(self):
        return self.atoms.calc.get_uncertainties()  

    def wrap_positions(self):
        return self.get_positions(wrap=True)

    @property
    def wrapped_positions(self):
        return self.get_positions(wrap=True)

    @property
    def max_cutoff(self):
        return get_max_cutoff(self.cell)
