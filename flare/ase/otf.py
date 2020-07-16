"""
:class:`OTF` is the on-the-fly training module for ASE, WITHOUT molecular dynamics engine. 
It needs to be used adjointly with ASE MD engine. 
"""
import os
import sys
import inspect
from time import time
from copy import deepcopy

import numpy as np
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units

from flare.otf import OTF
from flare.utils.learner import is_std_in_bound

from flare.ase.atoms import FLARE_Atoms
from flare.ase.calculator import FLARE_Calculator
import flare.ase.dft as dft_source


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
    refer to :class:`OTF`

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
        self.md_engine = md_engine

        if md_engine == "VelocityVerlet":
            MD = VelocityVerlet
        elif md_engine == "NVTBerendsen":
            MD = NVTBerendsen
        elif md_engine == "NPTBerendsen":
            MD = NPTBerendsen
        elif md_engine == "NPT":
            MD = NPT
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
            self.atoms.set_calculator(self.flare_calc)

        self.atoms.calc.calculate(self.atoms)

    def md_step(self):
        """
        Get new position in molecular dynamics based on the forces predicted by
        FLARE_Calculator or DFT calculator
        """
        # Update previous positions.
        self.structure.prev_positions = np.copy(self.structure.positions)

        # Take MD step.
        self.md.step()

        # Update the positions and cell of the structure object.
        self.structure.cell = np.copy(self.atoms.cell)
        self.structure.positions = np.copy(self.atoms.positions)

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

    def update_gp(self, train_atoms, dft_frcs):
        super().update_gp(train_atoms, dft_frcs)

        if self.flare_calc.use_mapping:
            self.flare_calc.mgp_model.build_map(self.flare_calc.gp_model)
