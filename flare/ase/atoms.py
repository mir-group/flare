import os
import sys
import inspect
from time import time
from copy import deepcopy

import numpy as np
from ase import Atoms
from ase.io import read, write
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

    @staticmethod
    def from_ase_atoms(atoms):
        """
        Args:
            atoms (ASE Atoms): the ase atoms to build from
        """
        kw = [
            "symbols",  # use either "symbols" or "number"
            "positions",
            "tags",
            "momenta",
            "masses",
            "magmoms",
            "charges",
            "scaled_positions",
            "cell",
            "pbc",
            "celldisp",
            "constraint",
            "calculator",
            "info",
            "velocities",
        ]
        # The keywords are either the attr of atoms, or in dict atoms.arrays
        kwargs = {}
        for key in kw:
            try:
                kwargs[key] = getattr(atoms, key)
            except:
                if key in atoms.arrays:
                    kwargs[key] = atoms.arrays[key]
        kwargs["calculator"] = atoms.calc
        return FLARE_Atoms(**kwargs)

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
        return self.get_forces()

    @forces.setter
    def forces(self, forces_array):
        pass

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
