import os
import sys
import inspect
from time import time
from copy import deepcopy

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from flare.utils.learner import get_max_cutoff


class FLARE_Atoms(Atoms):
    """
    The `FLARE_Atoms` class is a child class of ASE `Atoms`,
    which has completely the same usage as the primitive ASE `Atoms`, and
    in the meanwhile mimic `Structure` class. It is used in the `OTF` module
    with ASE engine (by `OTF_ASE` module). It enables attributes to be
    obtained by both the name from ASE `Atoms` and `Structure`.

    The input arguments are the same as ASE `Atoms`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_positions = np.zeros_like(self.positions)
        self.pbc = True # by default set periodic boundary

    @staticmethod
    def from_ase_atoms(atoms):
        """
        Args:
            atoms (ASE Atoms): the ase atoms to build from
        """
        new_atoms = deepcopy(atoms)
        new_atoms.__class__ = FLARE_Atoms
        new_atoms.prev_positions = np.zeros_like(new_atoms.positions)
        return new_atoms

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
        if self.calc is not None:
            return self.get_forces()
        else:
            return None

    @forces.setter
    def forces(self, forces_array):
        assert forces_array.shape[0] == len(self)
        assert forces_array.shape[1] == 3
        if self.calc is None: # attach calculator if there is none
            calc = SinglePointCalculator(self, forces=forces_array)
            self.calc = calc
        else: # update the forces in the calculator
            self.calc.results["forces"] = forces_array

    @property
    def potential_energy(self):
        return self.get_potential_energy()

    @property
    def stress(self):
        return self.get_stress()

    @property
    def stress_stds(self):
        return None  # TODO: to implement

    @property
    def local_energy_stds(self):
        return None  # TODO: to implement

    @property
    def stds(self):
        try:  # when self.calc is not FLARE, there's no get_uncertainties()
            stds = self.calc.results["stds"]
        except:
            stds = np.zeros_like(self.positions)
        return stds

    def wrap_positions(self):
        return self.get_positions(wrap=True)

    @property
    def wrapped_positions(self):
        return self.get_positions(wrap=True)

    @property
    def max_cutoff(self):
        return get_max_cutoff(self.cell)

    def as_dict(self):
        return self.todict()

    @staticmethod
    def from_dict(dct):
        atoms = Atoms.fromdict(dct)
        return FLARE_Atoms.from_ase_atoms(atoms)
