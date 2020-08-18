"""
:class:`ASE_OTF` is the on-the-fly training module for ASE, WITHOUT molecular dynamics engine. 
It needs to be used adjointly with ASE MD engine. 
"""
import os
import sys
import inspect
import pickle
from time import time
from copy import deepcopy
import logging

import numpy as np
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from ase.io import read, write

import flare
from flare.otf import OTF
from flare.utils.learner import is_std_in_bound

from flare.ase.atoms import FLARE_Atoms
from flare.ase.calculator import FLARE_Calculator
import flare.ase.dft as dft_source


def reset_npt_momenta(npt_engine, force):
    # in the last step, the momenta was set by flare forces, change to dft forces
    npt_engine._calculate_q_future(force)
    npt_engine.atoms.set_momenta(
        np.dot(
            npt_engine.q_future - npt_engine.q_past, npt_engine.h / (2 * npt_engine.dt)
        )
        * npt_engine._getmasses()
    )


class ASE_OTF(OTF):

    """
    On-the-fly training module using ASE MD engine, a subclass of OTF.

    Args:
        atoms (ASE Atoms): the ASE Atoms object for the on-the-fly MD run, 
            with calculator set as FLARE_Calculator.
        timestep: the timestep in MD. Please use ASE units, e.g. if the
            timestep is 1 fs, then set `timestep = 1 * units.fs`
        number_of_steps (int): the total number of steps for MD.
        dft_calc (ASE Calculator): any ASE calculator is supported, 
            e.g. Espresso, VASP etc.
        md_engine (str): the name of MD thermostat, only `VelocityVerlet`,
            `NVTBerendsen`, `NPTBerendsen`, `NPT` and `Langevin` are supported.
        md_kwargs (dict): Specify the args for MD as a dictionary, the args are
            as required by the ASE MD modules consistent with the `md_engine`.
        trajectory (ASE Trajectory): default `None`, not recommended,
            currently in experiment.

    The following arguments are for on-the-fly training, the user can also
    refer to :class:`flare.otf.OTF`

    Args:
        prev_pos_init ([type], optional): Previous positions. Defaults
            to None.
        rescale_steps (List[int], optional): List of frames for which the
            velocities of the atoms are rescaled. Defaults to [].
        rescale_temps (List[int], optional): List of rescaled temperatures.
            Defaults to [].

        calculate_energy (bool, optional): If True, the energy of each
            frame is calculated with the GP. Defaults to False.
        write_model (int, optional): If 0, write never. If 1, write at
            end of run. If 2, write after each training and end of run.
            If 3, write after each time atoms are added and end of run.

        std_tolerance_factor (float, optional): Threshold that determines
            when DFT is called. Specifies a multiple of the current noise
            hyperparameter. If the epistemic uncertainty on a force
            component exceeds this value, DFT is called. Defaults to 1.
        skip (int, optional): Number of frames that are skipped when
            dumping to the output file. Defaults to 0.
        init_atoms (List[int], optional): List of atoms from the input
            structure whose local environments and force components are
            used to train the initial GP model. If None is specified, all
            atoms are used to train the initial GP. Defaults to None.
        output_name (str, optional): Name of the output file. Defaults to
            'otf_run'.
        max_atoms_added (int, optional): Number of atoms added each time
            DFT is called. Defaults to 1.
        freeze_hyps (int, optional): Specifies the number of times the
            hyperparameters of the GP are optimized. After this many
            updates to the GP, the hyperparameters are frozen.
            Defaults to 10.

        n_cpus (int, optional): Number of cpus used during training.
            Defaults to 1.
    """

    def __init__(
        self,
        atoms,
        timestep,
        number_of_steps,
        dft_calc,
        md_engine,
        md_kwargs,
        trajectory=None,
        **otf_kwargs
    ):

        self.atoms = FLARE_Atoms.from_ase_atoms(atoms)
        self.timestep = timestep
        self.md_engine = md_engine
        self.md_kwargs = md_kwargs

        if md_engine == "VelocityVerlet":
            MD = VelocityVerlet
        elif md_engine == "NVTBerendsen":
            MD = NVTBerendsen
        elif md_engine == "NPTBerendsen":
            MD = NPTBerendsen
        elif md_engine == "NPT":
            MD = NPT
            # TODO: solve the md step
            assert (
                md_kwargs["pfactor"] is None
            ), "Current MD OTF only supports pfactor=None"
        elif md_engine == "Langevin":
            MD = Langevin
        else:
            raise NotImplementedError(md_engine + " is not implemented in ASE")

        self.md = MD(
            atoms=self.atoms, timestep=timestep, trajectory=trajectory, **md_kwargs
        )

        force_source = dft_source
        self.flare_calc = self.atoms.calc

        # Convert ASE timestep to ps for the output file.
        flare_dt = timestep / (units.fs * 1e3)

        super().__init__(
            dt=flare_dt,
            number_of_steps=number_of_steps,
            gp=self.flare_calc.gp_model,
            force_source=force_source,
            dft_loc=dft_calc,
            dft_input=self.atoms,
            **otf_kwargs
        )

        self.flare_name = self.output_name + "_flare.json"
        self.dft_name = self.output_name + "_dft.pickle"
        self.atoms_name = self.output_name + "_atoms.json"

    def get_structure_from_input(self, prev_pos_init):
        self.structure = self.atoms
        if prev_pos_init is None:
            self.atoms.prev_positions = np.copy(self.atoms.positions)
        else:
            assert len(self.atoms.positions) == len(
                self.atoms.prev_positions
            ), "Previous positions and positions are not same length"
            self.atoms.prev_positions = prev_pos_init

    def initialize_train(self):
        super().initialize_train()

        if self.md_engine == "NPT":
            if not self.md.initialized:
                self.md.initialize()
            else:
                if self.md.have_the_atoms_been_changed():
                    raise NotImplementedError(
                        "You have modified the atoms since the last timestep."
                    )

    def compute_properties(self):
        """
        Compute energies, forces, stresses, and their uncertainties with
            the FLARE ASE calcuator, and write the results to the
            OTF structure object.
        """

        # Reset FLARE calculator if necessary.
        if not isinstance(self.atoms.calc, FLARE_Calculator):
            self.atoms.calc = self.flare_calc

        if not self.flare_calc.results:
            self.atoms.calc.calculate(self.atoms)

    def md_step(self):
        """
        Get new position in molecular dynamics based on the forces predicted by
        FLARE_Calculator or DFT calculator
        """
        # Update previous positions.
        self.structure.prev_positions = np.copy(self.structure.positions)

        # Take MD step.
        f = self.atoms.get_forces()

        if self.md_engine == "NPT":
            self.flare_calc.results = {}  # init flare calculator

            if self.dft_step:
                reset_npt_momenta(self.md, f)
                self.atoms.calc = self.flare_calc

            self.md.step()  # use flare to get force for next step
        else:
            self.flare_calc.results = {}  # init flare calculator
            if self.dft_step:
                self.atoms.calc = self.flare_calc

            self.md.step(f)

        # Update the positions and cell of the structure object.
        self.structure.cell = np.copy(self.atoms.cell)
        self.structure.positions = np.copy(self.atoms.positions)

    def write_gp(self):
        self.flare_calc.write_model(self.flare_name)

    def update_positions(self, new_pos):
        # call OTF method
        super().update_positions(new_pos)

        # update ASE atoms
        if self.curr_step in self.rescale_steps:
            rescale_ind = self.rescale_steps.index(self.curr_step)
            temp_fac = self.rescale_temps[rescale_ind] / self.temperature
            vel_fac = np.sqrt(temp_fac)
            curr_velocities = self.atoms.get_velocities()
            self.atoms.set_velocities(curr_velocities * vel_fac)

    def update_temperature(self):
        self.KE = self.atoms.get_kinetic_energy()
        self.temperature = self.atoms.get_temperature()

        # Convert velocities to Angstrom / ps.
        self.velocities = self.atoms.get_velocities() * units.fs * 1e3

    def update_gp(self, train_atoms, dft_frcs, dft_energy=None, dft_stress=None):
        self.output.add_atom_info(train_atoms, self.structure.stds)

        # update gp model
        self.gp.update_db(
            self.structure,
            dft_frcs,
            custom_range=train_atoms,
            energy=dft_energy,
            stress=dft_stress,
        )

        self.gp.set_L_alpha()

        # train model
        if (self.dft_count - 1) < self.freeze_hyps:
            self.train_gp()

        # update mgp model
        if self.flare_calc.use_mapping:
            self.flare_calc.mgp_model.build_map(self.flare_calc.gp_model)

        # write model
        if (self.dft_count - 1) < self.freeze_hyps:
            if self.write_model == 2:
                self.write_gp()
        if self.write_model == 3:
            self.write_gp()

    def as_dict(self):

        # DFT module and Trajectory will cause issue in deepcopy
        self.dft_module = self.dft_module.__name__
        md = self.md
        self.md = None

        dct = deepcopy(dict(vars(self)))
        self.dft_module = eval(self.dft_module)
        self.md = md

        # write atoms and flare calculator to separate files
        write(self.atoms_name, self.atoms)
        dct["atoms"] = self.atoms_name

        self.flare_calc.write_model(self.flare_name)
        dct["flare_calc"] = self.flare_name

        # dump dft calculator as pickle
        with open(self.dft_name, "wb") as f:
            pickle.dump(self.dft_loc, f)  # dft_loc is the dft calculator
        dct["dft_loc"] = self.dft_name

        dct["gp"] = self.gp_name

        for key in ["output", "pred_func", "structure", "dft_input", "md"]:
            dct.pop(key)

        return dct

    @staticmethod
    def from_dict(dct):
        flare_calc = FLARE_Calculator.from_file(dct["flare_calc"])
        dct["atoms"] = read(dct["atoms"])
        dct["atoms"].calc = flare_calc
        dct.pop("gp")

        with open(dct["dft_loc"], "rb") as f:
            dct["dft_calc"] = pickle.load(f)

        for key in ["dt", "dft_loc"]:
            dct.pop(key)

        new_otf = ASE_OTF(**dct)
        new_otf.dft_count = dct["dft_count"]
        new_otf.curr_step = dct["curr_step"]

        if new_otf.md_engine == "NPT":
            if not new_otf.md.initialized:
                new_otf.md.initialize()

        return new_otf
