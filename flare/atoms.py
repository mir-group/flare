import os
import sys
import inspect
from typing import List, Union, Any
from abc import abstractmethod
from time import time
from copy import deepcopy

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from flare.learners.utils import get_max_cutoff


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
        self.pbc = True  # by default set periodic boundary

    @staticmethod
    def from_ase_atoms(atoms, copy_calc_results=False):
        """
        Args:
            atoms (ASE Atoms): the ase atoms to build from
        """
        new_atoms = deepcopy(atoms)
        new_atoms.__class__ = FLARE_Atoms
        new_atoms.prev_positions = np.zeros_like(new_atoms.positions)
        new_atoms.pbc = True
        if copy_calc_results:
            new_atoms.calc.results = deepcopy(atoms.calc.results)
        return new_atoms

    @property
    def nat(self):
        return len(self)

    @property
    def forces(self):
        if self.calc is not None:
            return self.get_forces()
        else:
            return np.zeros_like(self.positions)

    @forces.setter
    def forces(self, forces_array):
        assert forces_array.shape[0] == len(self), (forces_array.shape[0], len(self))
        assert forces_array.shape[1] == 3
        self.label_setter("forces", forces_array)

    def label_setter(self, key, value):
        if self.calc is None:  # attach calculator if there is none
            results = {
                "forces": np.zeros((len(self), 3)),
                "energy": 0,
                "stress": np.zeros(6),
                "stds": np.zeros((len(self), 3)),
                "local_energy_stds": np.zeros(len(self)),
                "stress_stds": np.zeros(6),
            }
            results.update({key: value})
            calc = SinglePointCalculator(self)
            self.calc = calc
            self.calc.results = results
        else:  # update the results in the calculator
            self.calc.results[key] = value

    @property
    def potential_energy(self):
        return self.get_potential_energy()

    @potential_energy.setter
    def potential_energy(self, energy_label):
        self.label_setter("energy", energy_label)

    @property
    def energy(self):
        try:
            return self.get_potential_energy()
        except:
            return 0

    @energy.setter
    def energy(self, energy_label):
        self.label_setter("energy", energy_label)

    @property
    def stress(self):
        return self.get_stress()

    @stress.setter
    def stress(self, stress_array):
        if (stress_array is None) or (len(stress_array) == 6):
            self.label_setter("stress", stress_array)
        elif len(stress_array) == 0:
            self.label_setter("stress", None)
        else:
            raise ValueError("stress_array should be None or array of length 6")

    @property
    def stress_stds(self):
        return None  # TODO: to implement

    @stress_stds.setter
    def stress_stds(self, stress_stds_label):
        self.label_setter("stress_stds", stress_stds_label)

    @property
    def local_energy_stds(self):
        return None  # TODO: to implement

    @local_energy_stds.setter
    def local_energy_stds(self, local_en_stds_label):
        self.label_setter("local_energy_stds", local_en_stds_label)

    @property
    def stds(self):
        try:  # when self.calc is not FLARE, there's no get_uncertainties()
            stds = self.calc.results["stds"]
        except:
            stds = np.zeros_like(self.positions)
        return stds

    @stds.setter
    def stds(self, stds_label):
        self.label_setter("stds", stds_label)

    def wrap_positions(self):
        return self.get_positions(wrap=True)

    @property
    def wrapped_positions(self):
        return self.get_positions(wrap=True)

    @property
    def max_cutoff(self):
        return get_max_cutoff(self.cell)

    def indices_of_specie(self, specie: Union[int, str]) -> List[int]:
        """
        Return the indices of a given species within atoms of the structure.

        :param specie: Element to target, can be string or integer
        :return: The indices in the structure at which this element occurs
        :rtype: List[str]
        """
        return [i for i, spec in enumerate(self.numbers) if spec == specie]

    def as_dict(self):
        dct = self.todict()
        if self.calc is not None:
            dct["results"] = self.calc.results

        return dct

    @staticmethod
    def from_dict(dct):
        if "results" in dct:
            results = dct.pop("results")
            results["forces"] = np.array(results["forces"])
            results["stress"] = np.array(results["stress"])
        else:
            results = None

        atoms = Atoms.fromdict(dct)
        if results is not None:
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = results

        return FLARE_Atoms.from_ase_atoms(atoms)


class StructureSource(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_next_structure(self) -> FLARE_Atoms:
        raise NotImplementedError

    def write_file(self):
        raise NotImplementedError


class ForceSource(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_next_force(self, *args, **kwargs) -> "np.ndarray":
        raise NotImplementedError

    def pre_force(self, *args, **kwargs):
        raise NotImplementedError


class Trajectory(StructureSource, ForceSource):
    def __init__(
        self, frames: List[FLARE_Atoms] = None, iterate_strategy: Union[int, str] = 1
    ):
        if frames is None:
            frames = []
        self.frames = frames
        self.cur_idx = 0
        self.iterate_strategy = iterate_strategy
        self.seen_before = []

        if self.iterate_strategy == "shuffle":
            self.frames = np.random.shuffle(frames)
        if isinstance(iterate_strategy, int):
            self.frames = frames[::iterate_strategy]

    def get_next_structure(self) -> Union[FLARE_Atoms, None]:

        if self.cur_idx == len(self):
            self.cur_idx = 0
            return None

        cur_frame = self.frames[self.cur_idx]
        self.cur_idx += 1
        return cur_frame

    @property
    def cur_frame(self):
        return self.frames[self.cur_idx]

    @property
    def cur_forces(self):
        return self.frames[self.cur_idx].forces

    def get_next_force(self, index: int = -1) -> "np.ndarray":
        """
        Return the forces associated with a current structure,
        and if an index is passed, that atom.
        :param index:
        :return:
        """
        if index != 1:
            return self.cur_frame.forces[index]
        return self.cur_frame.forces

    def __getitem__(self, item):
        return self.frames[item]

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        struc = self.get_next_structure()
        if struc is None:
            raise StopIteration()
        return struc

    def __iter__(self):
        self.cur_idx = 0
        return self

    def append(self, frame: FLARE_Atoms):
        self.frames.append(frame)
