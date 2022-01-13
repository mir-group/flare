"""
:class:`OTF` is the on-the-fly training module for ASE, WITHOUT molecular dynamics engine. 
It needs to be used adjointly with ASE MD engine. 
"""
import os
import sys
import pickle
import logging
import json
import time
import warnings
from copy import deepcopy
from datetime import datetime
from shutil import copyfile
from typing import List, Tuple, Union

import numpy as np

from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from flare.ase.npt import NPT_mod
from flare.ase.nosehoover import NoseHoover
from ase import units
from ase.io import read, write

import flare.predict as predict
from flare import gp
from flare.output import Output
from flare.utils.learner import is_std_in_bound
from flare.utils.element_coder import NumpyEncoder
from flare.ase.atoms import FLARE_Atoms
from flare.ase.calculator import FLARE_Calculator


class OTF:
    """Trains a Gaussian process force field on the fly during
        molecular dynamics.

    Args:
        atoms (ASE Atoms): the ASE Atoms object for the on-the-fly MD run.
        flare_calc: ASE calculator. Must have "get_uncertainties" method
          implemented.
        dt: the timestep in MD, in the units of pico-second. 
        number_of_steps (int): the total number of steps for MD.
        dft_calc (ASE Calculator): any ASE calculator is supported,
            e.g. Espresso, VASP etc.
        md_engine (str): the name of MD thermostat, only `VelocityVerlet`,
            `NVTBerendsen`, `NPTBerendsen`, `NPT` and `Langevin`, `NoseHoover`
            are supported.
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

        write_model (int, optional): If 0, write never. If 1, write at
            end of run. If 2, write after each training and end of run.
            If 3, write after each time atoms are added and end of run.
        force_only (bool, optional): If True, only use forces for training.
            Default to False, use forces, energy and stress for training.

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
        min_steps_with_model (int, optional): Minimum number of steps the
            model takes in between calls to DFT. Defaults to 0.
        dft_kwargs ([type], optional): Additional arguments which are
            passed when DFT is called; keyword arguments vary based on the
            program (e.g. ESPRESSO vs. VASP). Defaults to None.
        store_dft_output (Tuple[Union[str,List[str]],str], optional):
            After DFT calculations are called, copy the file or files
            specified in the first element of the tuple to a directory
            specified as the second element of the tuple.
            Useful when DFT calculations are expensive and want to be kept
            for later use. The first element of the tuple can either be a
            single file name, or a list of several. Copied files will be
            prepended with the date and time with the format
            'Year.Month.Day:Hour:Minute:Second:'.

        n_cpus (int, optional): Number of cpus used during training.
            Defaults to 1.
    """

    def __init__(
        self,
        # ase otf args
        atoms,
        dt,
        number_of_steps,
        dft_calc,
        md_engine,
        md_kwargs,
        flare_calc=None,
        trajectory=None,
        # md args
        prev_pos_init: "ndarray" = None,
        rescale_steps: List[int] = [],
        rescale_temps: List[int] = [],
        # flare args
        write_model: int = 0,
        force_only: bool = True,
        # otf args
        std_tolerance_factor: float = 1,
        skip: int = 0,
        init_atoms: List[int] = None,
        output_name: str = "otf_run",
        max_atoms_added: int = 1,
        freeze_hyps: int = 10,
        min_steps_with_model: int = 0,
        update_style: str = "add_n",
        update_threshold: float = None,
        # dft args
        dft_kwargs=None,
        store_dft_output: Tuple[Union[str, List[str]], str] = None,
        # other args
        n_cpus: int = 1,
        **kwargs,
    ):

        self.atoms = FLARE_Atoms.from_ase_atoms(atoms)
        if flare_calc is not None:
            self.atoms.calc = flare_calc 
        self.md_engine = md_engine
        self.md_kwargs = md_kwargs

        if md_engine == "VelocityVerlet":
            MD = VelocityVerlet
        elif md_engine == "NVTBerendsen":
            MD = NVTBerendsen
        elif md_engine == "NPTBerendsen":
            MD = NPTBerendsen
        elif md_engine == "NPT":
            MD = NPT_mod
        elif md_engine == "Langevin":
            MD = Langevin
        elif md_engine == "NoseHoover":
            MD = NoseHoover
        else:
            raise NotImplementedError(md_engine + " is not implemented in ASE")

        timestep = dt * units.fs * 1e3 # convert pico-second to ASE timestep units
        self.md = MD(
            atoms=self.atoms, timestep=timestep, trajectory=trajectory, **md_kwargs
        )

        self.flare_calc = self.atoms.calc

        # set DFT
        self.dft_loc = dft_calc
        self.dft_step = True
        self.dft_count = 0

        # set md
        self.dt = dt
        self.number_of_steps = number_of_steps
        self.get_structure_from_input(prev_pos_init)  # parse input file
        self.noa = self.structure.positions.shape[0]
        self.rescale_steps = rescale_steps
        self.rescale_temps = rescale_temps

        # set flare
        self.gp = self.flare_calc.gp_model
        self.force_only = force_only

        # set otf
        self.std_tolerance = std_tolerance_factor
        self.skip = skip
        self.max_atoms_added = max_atoms_added
        self.freeze_hyps = freeze_hyps
        if init_atoms is None:  # set atom list for initial dft run
            self.init_atoms = [int(n) for n in range(self.noa)]
        else:
            self.init_atoms = init_atoms
        self.update_style = update_style
        self.update_threshold = update_threshold

        self.n_cpus = n_cpus  # set number of cpus for DFT runs
        self.min_steps_with_model = min_steps_with_model

        self.dft_kwargs = dft_kwargs
        self.store_dft_output = store_dft_output

        # other args
        self.atom_list = list(range(self.noa))
        self.curr_step = 0
        self.steps_since_dft = 0

        # set logger
        self.output = Output(output_name, always_flush=True)
        self.output_name = output_name

        self.checkpt_name = self.output_name + "_checkpt.json"
        self.flare_name = self.output_name + "_flare.json"
        self.dft_name = self.output_name + "_dft.pickle"
        self.atoms_name = self.output_name + "_atoms.json"

        self.write_model = write_model

    def run(self):
        """
        Performs an on-the-fly training run.

        If OTF has store_dft_output set, then the specified DFT files will
        be copied with the current date and time prepended in the format
        'Year.Month.Day:Hour:Minute:Second:'.
        """

        optional_dict = {"Restart": self.curr_step}
        self.output.write_header(
            str(self.gp),
            self.dt,
            self.number_of_steps,
            self.structure,
            self.std_tolerance,
            optional_dict,
        )

        counter = 0
        self.start_time = time.time()

        while self.curr_step < self.number_of_steps:
            # run DFT and train initial model if first step and DFT is on
            if (
                (self.curr_step == 0)
                and (self.std_tolerance != 0)
                and (len(self.gp.training_data) == 0)
            ):

                # Are the recorded forces from the GP or DFT in ASE OTF?
                # When DFT is called, ASE energy, forces, and stresses should
                # get updated.
                self.initialize_train()

            # after step 1, try predicting with GP model
            else:
                # compute forces and stds with GP
                self.dft_step = False
                self.compute_properties()

                # get max uncertainty atoms
                std_in_bound, target_atoms = is_std_in_bound(
                    self.std_tolerance,
                    self.gp.force_noise,
                    self.structure,
                    max_atoms_added=self.max_atoms_added,
                    update_style=self.update_style,
                    update_threshold=self.update_threshold,
                )

                if (not std_in_bound) and (
                    self.steps_since_dft > self.min_steps_with_model
                ):
                    # record GP forces
                    self.update_temperature()
                    self.record_state()
                    gp_frcs = deepcopy(self.structure.forces)

                    # run DFT and record forces
                    self.dft_step = True
                    self.steps_since_dft = 0
                    self.run_dft()
                    dft_frcs = deepcopy(self.structure.forces)
                    dft_stress = deepcopy(self.structure.stress)
                    dft_energy = self.structure.potential_energy

                    # run MD step & record the state
                    self.record_state()

                    # compute mae and write to output
                    self.compute_mae(gp_frcs, dft_frcs)

                    # add max uncertainty atoms to training set
                    self.update_gp(
                        target_atoms,
                        dft_frcs,
                        dft_stress=dft_stress,
                        dft_energy=dft_energy,
                    )

            # write gp forces
            if counter >= self.skip and not self.dft_step:
                self.update_temperature()
                self.record_state()
                counter = 0

            counter += 1
            # TODO: Reinstate velocity rescaling.
            self.md_step()  # update positions by Verlet
            self.steps_since_dft += 1
            self.rescale_temperature(self.structure.positions)

            self.curr_step += 1

            if self.write_model == 3:
                self.checkpoint()

        self.output.conclude_run()

        if self.write_model >= 1:
            self.write_gp()
            self.checkpoint()

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
        # call dft and update positions
        self.run_dft()
        dft_frcs = deepcopy(self.structure.forces)
        dft_stress = deepcopy(self.structure.stress)
        dft_energy = self.structure.potential_energy

        self.update_temperature()
        self.record_state()

        # make initial gp model and predict forces
        self.update_gp(
            self.init_atoms, dft_frcs, dft_stress=dft_stress, dft_energy=dft_energy
        )

        # TODO: Turn this into a "reset" method.
        if not isinstance(self.atoms.calc, FLARE_Calculator):
            self.flare_calc.reset()
            self.atoms.calc = self.flare_calc

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

        # Change to FLARE calculator if necessary.
        if not isinstance(self.atoms.calc, FLARE_Calculator):
            self.flare_calc.reset()
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

        # Reset FLARE calculator.
        if self.dft_step:
            self.flare_calc.reset()
            self.atoms.calc = self.flare_calc

        # Take MD step. Inside the step() function, get_forces() is called
        self.md.step()

    def write_gp(self):
        self.flare_calc.write_model(self.flare_name)

    def run_dft(self):
        """Calculates DFT forces on atoms in the current structure.

        If OTF has store_dft_output set, then the specified DFT files will
        be copied with the current date and time prepended in the format
        'Year.Month.Day:Hour:Minute:Second:'.

        Calculates DFT forces on atoms in the current structure."""

        f = logging.getLogger(self.output.basename + "log")
        f.info("\nCalling DFT...\n")

        # change from FLARE to DFT calculator
        calc = deepcopy(self.dft_loc)
        self.atoms.set_calculator(calc)
    
        # Calculate DFT energy, forces, and stress.
        # Note that ASE and QE stresses differ by a minus sign.
        forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        energy = self.atoms.get_potential_energy()

        self.structure.forces = forces

        # write wall time of DFT calculation
        self.dft_count += 1
        self.output.conclude_dft(self.dft_count, self.start_time)

        # Store DFT outputs in another folder if desired
        # specified in self.store_dft_output
        if self.store_dft_output is not None:
            dest = self.store_dft_output[1]
            target_files = self.store_dft_output[0]
            now = datetime.now()
            dt_string = now.strftime("%Y.%m.%d:%H:%M:%S:")
            if isinstance(target_files, str):
                to_copy = [target_files]
            else:
                to_copy = target_files
            for ofile in to_copy:
                copyfile(ofile, dest + "/" + dt_string + ofile)

    def update_gp(
        self,
        train_atoms: List[int],
        dft_frcs: "ndarray",
        dft_energy: float = None,
        dft_stress: "ndarray" = None,
    ):
        """
        Updates the current GP model.


        Args:
            train_atoms (List[int]): List of atoms whose local environments
                will be added to the training set.
            dft_frcs (np.ndarray): DFT forces on all atoms in the structure.
        """
        stds = self.flare_calc.results.get("stds", np.zeros_like(dft_frcs))
        self.output.add_atom_info(train_atoms, stds)

        # Convert ASE stress (xx, yy, zz, yz, xz, xy) to FLARE stress
        # (xx, xy, xz, yy, yz, zz).
        flare_stress = None
        if dft_stress is not None:
            flare_stress = -np.array(
                [
                    dft_stress[0],
                    dft_stress[5],
                    dft_stress[4],
                    dft_stress[1],
                    dft_stress[3],
                    dft_stress[2],
                ]
            )

        if self.force_only:
            dft_energy = None
            flare_stress = None

        # update gp model
        self.gp.update_db(
            self.structure,
            dft_frcs,
            custom_range=train_atoms,
            energy=dft_energy,
            stress=flare_stress,
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


    def train_gp(self):
        """Optimizes the hyperparameters of the current GP model."""

        self.gp.train(logger_name=self.output.basename + "hyps")

        hyps, labels = self.gp.hyps_and_labels
        if labels is None:
            labels = self.gp.hyp_labels

        self.output.write_hyps(
            labels,
            hyps,
            self.start_time,
            self.gp.likelihood,
            self.gp.likelihood_gradient,
            hyps_mask=self.gp.hyps_mask,
        )

    def compute_mae(self, gp_frcs, dft_frcs):
        mae = np.mean(np.abs(gp_frcs - dft_frcs))
        mac = np.mean(np.abs(dft_frcs))

        f = logging.getLogger(self.output.basename + "log")
        f.info(f"mean absolute error: {mae:.4f} eV/A")
        f.info(f"mean absolute dft component: {mac:.4f} eV/A")

    def rescale_temperature(self, new_pos: "ndarray"):
        """Change the previous positions to update the temperature

        Args:
            new_pos (np.ndarray): Positions of atoms in the next MD frame.
        """
        if self.curr_step in self.rescale_steps:
            rescale_ind = self.rescale_steps.index(self.curr_step)
            temp_fac = self.rescale_temps[rescale_ind] / self.temperature
            vel_fac = np.sqrt(temp_fac)
            self.structure.prev_positions = (
                new_pos - self.velocities * self.dt * vel_fac
            )

        # update ASE atoms
        if self.curr_step in self.rescale_steps:
            rescale_ind = self.rescale_steps.index(self.curr_step)
            new_temp = self.rescale_temps[rescale_ind]
            temp_fac = new_temp / self.temperature
            vel_fac = np.sqrt(temp_fac)
            curr_velocities = self.atoms.get_velocities()
            self.atoms.set_velocities(curr_velocities * vel_fac)

            # Reset thermostat parameters.
            if self.md_engine in ["NVTBerendsen", "NPTBerendsen", "NPT", "Langevin"]:
                self.md.set_temperature(temperature_K=new_temp)
                self.md_kwargs["temperature"] = new_temp * units.kB

    def update_temperature(self):
        """Updates the instantaneous temperatures of the system.
        """
        self.KE = self.atoms.get_kinetic_energy()
        self.temperature = self.atoms.get_temperature()

        # Convert velocities to Angstrom / ps.
        self.velocities = self.atoms.get_velocities() * units.fs * 1e3

    def record_state(self):
        self.output.write_md_config(
            self.dt,
            self.curr_step,
            self.structure,
            self.temperature,
            self.KE,
            self.start_time,
            self.dft_step,
            self.velocities,
        )

    def as_dict(self):
        # DFT module and Trajectory will cause issue in deepcopy
        md = self.md
        self.md = None

        # SGP models aren't picklable. Temporarily set to None before copying.
        flare_calc = self.flare_calc
        gp = self.gp
        self.flare_calc = None
        self.gp = None
        self.atoms.calc = None

        # Deepcopy OTF object.
        dct = deepcopy(dict(vars(self)))

        # Reset attributes.
        self.md = md
        self.flare_calc = flare_calc
        self.gp = gp
        self.atoms.calc = flare_calc

        # write atoms and flare calculator to separate files
        write(self.atoms_name, self.atoms)
        dct["atoms"] = self.atoms_name

        self.flare_calc.write_model(self.flare_name)
        dct["flare_calc"] = self.flare_name

        # dump dft calculator as pickle
        with open(self.dft_name, "wb") as f:
            pickle.dump(self.dft_loc, f)  # dft_loc is the dft calculator
        dct["dft_loc"] = self.dft_name

        for key in ["output", "structure", "md"]:
            dct.pop(key)

        return dct

    @staticmethod
    def from_dict(dct):
        dct["atoms"] = read(dct["atoms"])
        flare_calc = FLARE_Calculator.from_file(dct["flare_calc"])
        flare_calc.reset()
        dct["flare_calc"] = flare_calc

        with open(dct["dft_loc"], "rb") as f:
            dct["dft_calc"] = pickle.load(f)

        for key in ["dft_loc"]:
            dct.pop(key)

        new_otf = OTF(**dct)
        new_otf.dft_count = dct["dft_count"]
        new_otf.curr_step = dct["curr_step"]
        new_otf.std_tolerance = dct["std_tolerance"]

        if new_otf.md_engine == "NPT":
            if not new_otf.md.initialized:
                new_otf.md.initialize()

        return new_otf

    def checkpoint(self):
        name = self.checkpt_name
        if ".json" != name[-5:]:
            name += ".json"
        with open(name, "w") as f:
            json.dump(self.as_dict(), f, cls=NumpyEncoder)

    @classmethod
    def from_checkpoint(cls, filename):
        with open(filename, "r") as f:
            otf_model = cls.from_dict(json.loads(f.readline()))

        return otf_model
