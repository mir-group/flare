"""
:class:`OTF` is the on-the-fly training module for ASE, WITHOUT molecular dynamics engine. 
It needs to be used adjointly with ASE MD engine. 
"""
import os
import sys
import pickle
import logging
import os
import shutil
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
from flare.md.npt import NPT_mod
from flare.md.nosehoover import NoseHoover
from flare.md.lammps import LAMMPS_MD, check_sgp_match
from flare.md.fake import FakeMD
from ase import units
from ase.io import read, write

from flare.io.output import Output
from flare.learners.utils import is_std_in_bound, get_env_indices
from flare.utils import NumpyEncoder
from flare.atoms import FLARE_Atoms
from flare.bffs.gp.calculator import FLARE_Calculator


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
            If 4, write after each training and end of run, and back up
            after each write.
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
        build_mode (str): default "bayesian", run on-the-fly training.
            "direct" mode constructs GP model from a given list of frames, with
            `FakeMD` and `FakeDFT`. Each frame needs to have a global 
            property called "target_atoms" specifying a list of atomic
            environments added to the GP model.
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
        build_mode="bayesian",
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
        elif md_engine == "PyLAMMPS":
            MD = LAMMPS_MD
        elif md_engine == "Fake":
            MD = FakeMD
        else:
            raise NotImplementedError(md_engine + " is not implemented in ASE")

        timestep = dt * units.fs * 1e3 # convert pico-second to ASE timestep units
        self.md = MD(
            atoms=self.atoms, timestep=timestep, trajectory=trajectory, **md_kwargs
        )

        self.flare_calc = self.atoms.calc

        # set DFT
        self.dft_calc = dft_calc
        self.dft_step = True
        self.dft_count = 0

        # set md
        self.dt = dt
        self.number_of_steps = number_of_steps
        self.get_structure_from_input(prev_pos_init)  # parse input file
        self.noa = len(self.atoms)
        self.rescale_steps = rescale_steps
        self.rescale_temps = rescale_temps

        # set flare
        self.gp = self.flare_calc.gp_model
        self.force_only = force_only
        self._kernels = None

        # set otf
        self.std_tolerance = std_tolerance_factor
        self.skip = skip
        if max_atoms_added < 0:
            self.max_atoms_added = self.noa
        else:
            self.max_atoms_added = max_atoms_added
        self.freeze_hyps = freeze_hyps
        if init_atoms is None:  # set atom list for initial dft run
            self.init_atoms = [int(n) for n in range(self.noa)]
        else:
            # detect if there are duplicated atoms 
            assert len(set(init_atoms)) == len(init_atoms), \
                    "init_atoms should not include duplicated indices"
            self.init_atoms = init_atoms

        self.update_style = update_style
        self.update_threshold = update_threshold

        self.min_steps_with_model = min_steps_with_model

        self.dft_kwargs = dft_kwargs
        self.store_dft_output = store_dft_output

        # other args
        self.atom_list = list(range(self.noa))
        self.curr_step = 0
        self.last_dft_step = 0
        self.build_mode = build_mode

        if self.build_mode not in ["bayesian", "direct"]:
            raise Exception("build_mode needs to be 'bayesian' or 'direct'")

        # set logger
        self.output = Output(output_name, always_flush=True, print_as_xyz=True)
        self.output_name = output_name

        self.checkpt_name = self.output_name + "_checkpt.json"
        self.flare_name = self.output_name + "_flare.json"
        self.dft_name = self.output_name + "_dft.pickle"
        self.atoms_name = self.output_name + "_atoms.json"
        self.dft_xyz = self.output_name + "_dft.xyz"
        self.checkpt_files = [
            self.checkpt_name,
            self.flare_name,
            self.dft_name,
            self.atoms_name,
            self.dft_xyz,
        ]

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
            self.atoms,
            self.std_tolerance,
            optional_dict,
        )

        counter = 0
        self.start_time = time.time()
        exit_flag = False

        while (self.curr_step < self.number_of_steps) and (not exit_flag):
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
                self.initialize_md()

            # starting MD with a non-empty GP
            elif (self.curr_step == 0) and (len(self.gp.training_data) > 0):
                self.initialize_md()

            # after step 1, try predicting with GP model
            else:
                # compute forces and stds with GP
                self.dft_step = False
                self.compute_properties()

                # get max uncertainty atoms
                if self.build_mode == "bayesian":
                    env_selection = is_std_in_bound
                elif self.build_mode == "direct":
                    env_selection = get_env_indices

                std_in_bound, target_atoms = env_selection(
                    self.std_tolerance,
                    self.gp.force_noise,
                    self.atoms,
                    max_atoms_added=self.max_atoms_added,
                    update_style=self.update_style,
                    update_threshold=self.update_threshold,
                )

                steps_since_dft = self.curr_step - self.last_dft_step
                if (not std_in_bound) and (steps_since_dft > self.min_steps_with_model):
                    # record GP forces
                    self.update_temperature()
                    self.record_state()

                    gp_energy = self.atoms.potential_energy
                    gp_forces = deepcopy(self.atoms.forces)
                    gp_stress = deepcopy(self.atoms.stress)

                    # run DFT and record forces
                    self.dft_step = True
                    self.last_dft_step = self.curr_step
                    self.run_dft()

                    dft_frcs = deepcopy(self.atoms.forces)
                    dft_stress = deepcopy(self.atoms.stress)
                    dft_energy = self.atoms.potential_energy

                    # run MD step & record the state
                    self.record_state()

                    # record DFT data into an .xyz file with filename self.dft_xyz.
                    # the file includes the structure, e/f/s labels and atomic
                    # indices of environments added to gp
                    self.record_dft_data(self.atoms, target_atoms)

                    # compute mae and write to output
                    self.compute_mae(
                        gp_energy,
                        gp_forces,
                        gp_stress,
                        dft_energy,
                        dft_frcs,
                        dft_stress,
                    )

                    # add max uncertainty atoms to training set
                    self.update_gp(
                        target_atoms,
                        dft_frcs,
                        dft_stress=dft_stress,
                        dft_energy=dft_energy,
                    )

                    if self.write_model == 4:
                        self.checkpoint()
                        self.backup_checkpoint()

            # write gp forces
            if counter >= self.skip and not self.dft_step:
                self.update_temperature()
                self.record_state()
                counter = 0

            counter += 1
            # TODO: Reinstate velocity rescaling.
            step_status = self.md_step()  # update positions by Verlet
            exit_flag = step_status == 1
            self.rescale_temperature(self.atoms.positions)

            if self.write_model == 3:
                self.checkpoint()

        self.output.conclude_run()

        if self.write_model >= 1:
            self.write_gp()
            self.checkpoint()

    def get_structure_from_input(self, prev_pos_init):
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
        dft_frcs = deepcopy(self.atoms.forces)
        dft_stress = deepcopy(self.atoms.stress)
        dft_energy = self.atoms.potential_energy

        self.update_temperature()
        self.record_state()
        self.record_dft_data(self.atoms, self.init_atoms)

        # make initial gp model and predict forces
        self.update_gp(
            self.init_atoms, dft_frcs, dft_stress=dft_stress, dft_energy=dft_energy
        )

    def initialize_md(self):
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
        self.atoms.prev_positions = np.copy(self.atoms.positions)

        # Reset FLARE calculator.
        if self.dft_step:
            self.flare_calc.reset()
            self.atoms.calc = self.flare_calc

        # Take MD step.
        if self.md_engine == "PyLAMMPS":
            if self.std_tolerance < 0:
                tol = -self.std_tolerance
            else:
                tol = np.abs(self.gp.force_noise) * self.std_tolerance
            f = logging.getLogger(self.output.basename + "log")
            step_status = self.md.step(tol, self.number_of_steps)
            self.curr_step = self.md.nsteps
            self.atoms = FLARE_Atoms.from_ase_atoms(self.md.curr_atoms)

            # check if the lammps energy/forces/stress/stds match sgp
            f = logging.getLogger(self.output.basename + "log")
            check_sgp_match(
                self.atoms, self.flare_calc, f, self.md.params["specorder"], self.md.command,
            )

        else:
            # Inside the step() function, get_forces() is called
            step_status = self.md.step()
            self.curr_step += 1
        return step_status

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
        self.atoms.calc = deepcopy(self.dft_calc)
    
        # Calculate DFT energy, forces, and stress.
        # Note that ASE and QE stresses differ by a minus sign.
        forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        energy = self.atoms.get_potential_energy()

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
                # if the file is in a subdirectory like dft/OUTCAR, then copy it out
                filename = ofile.split("/")[-1]
                copyfile(ofile, dest + "/" + dt_string + filename)

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

        # The structure will be added to self.gp.training_structures (FLARE_Atoms).
        # Create a new structure by deepcopy to avoid the forces of the saved
        # structure get modified.
        try:
            struc_to_add = deepcopy(self.atoms)
        except TypeError:
            # The structure might be attached with a non-picklable calculator,
            # e.g., when we use LAMMPS empirical potential for training. 
            # When deepcopy fails, create a SinglePointCalculator to store results

            from ase.calculators.singlepoint import SinglePointCalculator

            properties = ["forces", "energy", "stress"]
            results = {
                "forces": self.atoms.forces,
                "energy": self.atoms.potential_energy,
                "stress": self.atoms.stress,
            }

            calc = self.atoms.calc
            self.atoms.calc = None
            struc_to_add = deepcopy(self.atoms)
            struc_to_add.calc = SinglePointCalculator(struc_to_add, **results)
            self.atoms.calc = calc

        # update gp model
        self.gp.update_db(
            struc_to_add,
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
            self.flare_calc.build_map()

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

    def compute_mae(
        self,
        gp_energy,
        gp_forces,
        gp_stress,
        dft_energy,
        dft_forces,
        dft_stress,
    ):

        f = logging.getLogger(self.output.basename + "log")
        f.info("Mean absolute errors & Mean absolute values")

        # compute energy/forces/stress mean absolute error and value
        if not self.force_only:
            e_mae = np.mean(np.abs(dft_energy - gp_energy))
            e_mav = np.mean(np.abs(dft_energy))
            f.info(f"energy mae: {e_mae:.4f} eV")
            f.info(f"energy mav: {e_mav:.4f} eV")

            s_mae = np.mean(np.abs(dft_stress - gp_stress))
            s_mav = np.mean(np.abs(dft_stress))
            f.info(f"stress mae: {s_mae:.4f} eV/A^3")
            f.info(f"stress mav: {s_mav:.4f} eV/A^3")

        f_mae = np.mean(np.abs(dft_forces - gp_forces))
        f_mav = np.mean(np.abs(dft_forces))
        f.info(f"forces mae: {f_mae:.4f} eV/A")
        f.info(f"forces mav: {f_mav:.4f} eV/A")

        # compute the per-species MAE
        unique_species = list(set(self.atoms.numbers))
        per_species_mae = np.zeros(len(unique_species))
        per_species_mav = np.zeros(len(unique_species))
        per_species_num = np.zeros(len(unique_species))
        for a in range(self.atoms.nat):
            species_ind = unique_species.index(self.atoms.numbers[a])
            per_species_mae[species_ind] += np.mean(
                np.abs(dft_forces[a] - gp_forces[a])
            )
            per_species_mav[species_ind] += np.mean(np.abs(dft_forces[a]))
            per_species_num[species_ind] += 1
        per_species_mae /= per_species_num
        per_species_mav /= per_species_num

        for s in range(len(unique_species)):
            curr_species = unique_species[s]
            f.info(f"type {curr_species} forces mae: {per_species_mae[s]:.4f} eV/A")
            f.info(f"type {curr_species} forces mav: {per_species_mav[s]:.4f} eV/A")

    def rescale_temperature(self, new_pos: "ndarray"):
        """Change the previous positions to update the temperature

        Args:
            new_pos (np.ndarray): Positions of atoms in the next MD frame.
        """
        if self.curr_step in self.rescale_steps:
            rescale_ind = self.rescale_steps.index(self.curr_step)
            temp_fac = self.rescale_temps[rescale_ind] / self.temperature
            vel_fac = np.sqrt(temp_fac)
            self.atoms.prev_positions = (
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
            self.atoms,
            self.temperature,
            self.KE,
            self.start_time,
            self.dft_step,
            self.velocities,
        )

    def record_dft_data(self, structure, target_atoms):
        structure.info["target_atoms"] = np.array(target_atoms)
        write(self.dft_xyz, structure, append=True)

    def as_dict(self):
        # DFT module and Trajectory are not picklable
        md = self.md
        self.md = None
        _kernels = self._kernels
        self._kernels = None

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
        self._kernels = _kernels
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
            pickle.dump(self.dft_calc, f)
        dct["dft_calc"] = self.dft_name

        for key in ["output", "md"]:
            dct.pop(key)

        if self.md_engine == "PyLAMMPS":
            dct["md_nsteps"] = self.md.nsteps

        return dct

    @staticmethod
    def from_dict(dct):
        flare_calc_dict = json.load(open(dct["flare_calc"]))

        # Build FLARE_Calculator from dict 
        if flare_calc_dict["class"] == "FLARE_Calculator":
            flare_calc = FLARE_Calculator.from_file(dct["flare_calc"])
            _kernels = None
        # Build SGP_Calculator from dict
        # TODO: we still have the issue that the c++ kernel needs to be 
        # in the current space, otherwise there is Seg Fault
        # That's why there is the _kernels
        elif flare_calc_dict["class"] == "SGP_Calculator":
            from flare.bffs.sgp.calculator import SGP_Calculator

            flare_calc, _kernels = SGP_Calculator.from_file(dct["flare_calc"])
        else:
            raise TypeError(
                f"The calculator from {dct['flare_calc']} is not recognized."
            )

        flare_calc.reset()
        dct["atoms"] = read(dct["atoms"])
        dct["flare_calc"] = flare_calc

        with open(dct["dft_calc"], "rb") as f:
            dct["dft_calc"] = pickle.load(f)

        new_otf = OTF(**dct)
        new_otf._kernels = _kernels
        new_otf.dft_count = dct["dft_count"]
        new_otf.curr_step = dct["curr_step"]
        new_otf.std_tolerance = dct["std_tolerance"]

        if new_otf.md_engine == "NPT":
            if not new_otf.md.initialized:
                new_otf.md.initialize()
        elif new_otf.md_engine == "PyLAMMPS":
            new_otf.md.nsteps = dct["md_nsteps"]
            assert new_otf.md.nsteps == new_otf.curr_step

        return new_otf

    def backup_checkpoint(self):
        dir_name = f"{self.output_name}_ckpt_{self.curr_step}"
        os.mkdir(dir_name)
        for f in self.checkpt_files:
            shutil.copyfile(f, f"{dir_name}/{f}")

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
