import os
import sys
import inspect
from time import time
from copy import deepcopy

import numpy as np
from ase import Atoms

class OTF_Atoms(Atoms):
    '''
    The `OTF_Atoms` class is a child class of ASE `Atoms`, 
    in the meanwhile mimic `Structure` class. It is used in the `OTF` module
    with ASE engine (by `OTF_ASE` module). It enables attributes to be 
    obtained by both the name from ASE `Atoms` and `Structure`.

    The input arguments are the same as ASE `Atoms`.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

    # TODO: See otf.py and output.py to add more properties

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
    def energy(self):
        return self.atoms.calc.get_potential_energy()

    @property
    def stress(self):
        return self.atoms.calc.get_stress()

    @property
    def stds(self):
        return self.atoms.calc.get_uncertainties()  
