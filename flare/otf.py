import sys
import numpy as np
import time
import copy
import multiprocessing as mp
import subprocess
from shutil import copyfile
from typing import List, Tuple, Union
from datetime import datetime

import flare.predict as predict
from flare import struc, gp, env, md
from flare.dft_interface import dft_software
from flare.output import Output
from flare.util import is_std_in_bound


class OTF:
    """Trains a Gaussian process force field on the fly during
        molecular dynamics.

    Args:
        dft_input (str): Input file.
        dt (float): MD timestep.
        number_of_steps (int): Number of timesteps in the training
            simulation.
        gp (gp.GaussianProcess): Initial GP model.
        dft_loc (str): Location of DFT executable.
        std_tolerance_factor (float, optional): Threshold that determines
            when DFT is called. Specifies a multiple of the current noise
            hyperparameter. If the epistemic uncertainty on a force
            component exceeds this value, DFT is called. Defaults to 1.
        prev_pos_init ([type], optional): Previous positions. Defaults
            to None.
        par (bool, optional): If True, force predictions are made in
            parallel. Defaults to False.
        skip (int, optional): Number of frames that are skipped when
            dumping to the output file. Defaults to 0.
        init_atoms (List[int], optional): List of atoms from the input
            structure whose local environments and force components are
            used to train the initial GP model. If None is specified, all
            atoms are used to train the initial GP. Defaults to None.
        calculate_energy (bool, optional): If True, the energy of each
            frame is calculated with the GP. Defaults to False.
        output_name (str, optional): Name of the output file. Defaults to
            'otf_run'.
        max_atoms_added (int, optional): Number of atoms added each time
            DFT is called. Defaults to 1.
        freeze_hyps (int, optional): Specifies the number of times the
            hyperparameters of the GP are optimized. After this many
            updates to the GP, the hyperparameters are frozen.
            Defaults to 10.
        rescale_steps (List[int], optional): List of frames for which the
            velocities of the atoms are rescaled. Defaults to [].
        rescale_temps (List[int], optional): List of rescaled temperatures.
            Defaults to [].
        force_source (Union[str, object], optional): DFT code used to calculate
            ab initio forces during training. A custom module can be used here
            in place of the DFT modules available in the FLARE package. The
            module must contain two functions: parse_dft_input, which takes a
            file name (in string format) as input and returns the positions,
            species, cell, and masses of a structure of atoms; and run_dft_par,
            which takes a number of DFT related inputs and returns the forces
            on all atoms.  Defaults to "qe".
        n_cpus (int, optional): Number of cpus used during training.
            Defaults to 1.
        npool (int, optional): Number of k-point pools for DFT
            calculations. Defaults to None.
        mpi (str, optional): Determines how mpi is called. Defaults to
            "srun".
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
    """
    def __init__(self, dft_input: str, dt: float, number_of_steps: int,
                 gp: gp.GaussianProcess, dft_loc: str,
                 std_tolerance_factor: float = 1,
                 prev_pos_init: 'ndarray' = None, par: bool = False,
                 skip: int = 0, init_atoms: List[int] = None,
                 calculate_energy: bool = False, output_name: str = 'otf_run',
                 max_atoms_added: int = 1, freeze_hyps: int = 10,
                 rescale_steps: List[int] = [], rescale_temps: List[int] = [],
                 force_source: Union[str, object] = "qe", n_cpus: int = 1,
                 npool: int = None, mpi: str = "srun", dft_kwargs=None,
                 store_dft_output: Tuple[Union[str, List[str]], str] = None):

        self.dft_input = dft_input
        self.dt = dt
        self.number_of_steps = number_of_steps
        self.gp = gp
        self.dft_loc = dft_loc
        self.std_tolerance = std_tolerance_factor
        self.skip = skip
        self.dft_step = True
        self.freeze_hyps = freeze_hyps

        if isinstance(force_source, str):
            self.dft_module = dft_software[force_source]
        else:
            self.dft_module = force_source

        # parse input file
        positions, species, cell, masses = \
            self.dft_module.parse_dft_input(self.dft_input)

        self.structure = struc.Structure(cell=cell, species=species,
                                         positions=positions,
                                         mass_dict=masses,
                                         prev_positions=prev_pos_init,
                                         species_labels=species)

        self.noa = self.structure.positions.shape[0]
        self.atom_list = list(range(self.noa))
        self.curr_step = 0

        self.max_atoms_added = max_atoms_added

        # initialize local energies
        if calculate_energy:
            self.local_energies = np.zeros(self.noa)
        else:
            self.local_energies = None

        # set atom list for initial dft run
        if init_atoms is None:
            self.init_atoms = [int(n) for n in range(self.noa)]
        else:
            self.init_atoms = init_atoms

        self.dft_count = 0

        # set pred function
        if (par and gp.per_atom_par and gp.par) and not calculate_energy:
            self.pred_func = predict.predict_on_structure_par
        elif not calculate_energy:
            self.pred_func = predict.predict_on_structure
        elif (par and gp.per_atom_par and gp.par):
            self.pred_func = predict.predict_on_structure_par_en
        else:
            self.pred_func = predict.predict_on_structure_en
        self.par = par

        # set rescale attributes
        self.rescale_steps = rescale_steps
        self.rescale_temps = rescale_temps

        self.output = Output(output_name, always_flush=True)

        # set number of cpus and npool for DFT runs
        self.n_cpus = n_cpus
        self.npool = npool
        self.mpi = mpi

        self.dft_kwargs = dft_kwargs
        self.store_dft_output = store_dft_output

    def run(self):
        """
        Performs an on-the-fly training run.

        If OTF has store_dft_output set, then the specified DFT files will
        be copied with the current date and time prepended in the format
        'Year.Month.Day:Hour:Minute:Second:'.
        """

        self.output.write_header(self.gp.cutoffs, self.gp.kernel_name,
                                 self.gp.hyps, self.gp.algo,
                                 self.dt, self.number_of_steps,
                                 self.structure,
                                 self.std_tolerance)
        counter = 0
        self.start_time = time.time()

        while self.curr_step < self.number_of_steps:
            # run DFT and train initial model if first step and DFT is on
            if self.curr_step == 0 and self.std_tolerance != 0 and len(self.gp.training_data)==0:
                # call dft and update positions
                self.run_dft()
                dft_frcs = copy.deepcopy(self.structure.forces)
                new_pos = md.update_positions(self.dt, self.noa,
                                              self.structure)
                self.update_temperature(new_pos)
                self.record_state()

                # make initial gp model and predict forces
                self.update_gp(self.init_atoms, dft_frcs)
                if (self.dft_count-1) < self.freeze_hyps:
                    self.train_gp()

            # after step 1, try predicting with GP model
            else:
                self.gp.check_L_alpha()
                self.pred_func(self.structure, self.gp, self.n_cpus)
                self.dft_step = False
                new_pos = md.update_positions(self.dt, self.noa,
                                              self.structure)

                # get max uncertainty atoms
                std_in_bound, target_atoms = \
                    is_std_in_bound(self.std_tolerance,
                                    self.gp.hyps[-1], self.structure,
                                    self.max_atoms_added)

                if not std_in_bound:
                    # record GP forces
                    self.update_temperature(new_pos)
                    self.record_state()
                    gp_frcs = copy.deepcopy(self.structure.forces)

                    # run DFT and record forces
                    self.dft_step = True
                    self.run_dft()
                    dft_frcs = copy.deepcopy(self.structure.forces)
                    new_pos = md.update_positions(self.dt, self.noa,
                                                  self.structure)
                    self.update_temperature(new_pos)
                    self.record_state()

                    # compute mae and write to output
                    mae = np.mean(np.abs(gp_frcs - dft_frcs))
                    mac = np.mean(np.abs(dft_frcs))

                    self.output.write_to_log('\nmean absolute error:'
                                             ' %.4f eV/A \n' % mae)
                    self.output.write_to_log('mean absolute dft component:'
                                             ' %.4f eV/A \n' % mac)

                    # add max uncertainty atoms to training set
                    self.update_gp(target_atoms, dft_frcs)
                    if (self.dft_count-1) < self.freeze_hyps:
                        self.train_gp()

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
                        for file in to_copy:
                            copyfile(file, dest+'/'+dt_string+file)

            # write gp forces
            if counter >= self.skip and not self.dft_step:
                self.update_temperature(new_pos)
                self.record_state()
                counter = 0

            counter += 1
            self.update_positions(new_pos)
            self.curr_step += 1

        self.output.conclude_run()

    def run_dft(self):
        """Calculates DFT forces on atoms in the current structure.

        If OTF has store_dft_output set, then the specified DFT files will
        be copied with the current date and time prepended in the format
        'Year.Month.Day:Hour:Minute:Second:'.

        Calculates DFT forces on atoms in the current structure."""

        self.output.write_to_log('\nCalling DFT...\n')

        # calculate DFT forces
        forces = self.dft_module.run_dft_par(self.dft_input, self.structure,
                                             self.dft_loc,
                                             n_cpus=self.n_cpus,
                                             npool=self.npool,
                                             mpi=self.mpi,
                                             dft_kwargs=self.dft_kwargs)
        self.structure.forces = forces

        # write wall time of DFT calculation
        self.dft_count += 1
        self.output.write_to_log('DFT run complete.\n')
        time_curr = time.time() - self.start_time
        self.output.write_to_log('number of DFT calls: %i \n' % self.dft_count)
        self.output.write_to_log('wall time from start: %.2f s \n' % time_curr)

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
            for file in to_copy:
                copyfile(file, dest+'/'+dt_string+file)

    def update_gp(self, train_atoms: List[int], dft_frcs: 'ndarray'):
        """
        Updates the current GP model.


        Args:
            train_atoms (List[int]): List of atoms whose local environments
                will be added to the training set.
            dft_frcs (np.ndarray): DFT forces on all atoms in the structure.
        """
        self.output.write_to_log('\nAdding atom {} to the training set.\n'
                                 .format(train_atoms))
        self.output.write_to_log('Uncertainty: {}.\n'
                                 .format(self.structure.stds[train_atoms[0]]))

        # update gp model
        self.gp.update_db(self.structure, dft_frcs,
                          custom_range=train_atoms)

        self.gp.set_L_alpha()

    def train_gp(self):
        """Optimizes the hyperparameters of the current GP model."""

        self.gp.train(self.output)
        self.output.write_hyps(self.gp.hyp_labels, self.gp.hyps,
                               self.start_time,
                               self.gp.likelihood, self.gp.likelihood_gradient)

    def update_positions(self, new_pos: 'ndarray'):
        """Performs a Verlet update of the atomic positions.

        Args:
            new_pos (np.ndarray): Positions of atoms in the next MD frame.
        """
        if self.curr_step in self.rescale_steps:
            rescale_ind = self.rescale_steps.index(self.curr_step)
            temp_fac = self.rescale_temps[rescale_ind] / self.temperature
            vel_fac = np.sqrt(temp_fac)
            self.structure.prev_positions = \
                new_pos - self.velocities * self.dt * vel_fac
        else:
            self.structure.prev_positions = self.structure.positions
        self.structure.positions = new_pos
        self.structure.wrap_positions()

    def update_temperature(self, new_pos: 'ndarray'):
        """Updates the instantaneous temperatures of the system.

        Args:
            new_pos (np.ndarray): Positions of atoms in the next MD frame.
        """
        KE, temperature, velocities = \
            md.calculate_temperature(new_pos, self.structure, self.dt,
                                     self.noa)
        self.KE = KE
        self.temperature = temperature
        self.velocities = velocities

    def record_state(self):
        self.output.write_md_config(self.dt, self.curr_step, self.structure,
                                    self.temperature, self.KE,
                                    self.local_energies, self.start_time,
                                    self.dft_step,
                                    self.velocities)
