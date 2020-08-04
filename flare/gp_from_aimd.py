"""
Tool to enable the development of a GP model based on an AIMD
trajectory with many customizable options for fine control of training.
Contains methods to transfer the model to an OTF run or MD engine run.


Seed frames
-----------
The various parameters in the :class:`TrajectoryTrainer` class related to
"Seed frames" are to help you  train a model which does not yet have a
training set. Uncertainty- and force-error driven training will go better with
a somewhat populated training set, as force and uncertainty estimates
are better behaveed with more data.

You may pass in a set of seed frames or atomic environments.
All seed environments will be added to the GP model; seed frames will
be iterated through and atoms will be added at random.
There are a few reasons why you would want to pay special attention to an
individual species.

If you are studying a system where the dynamics of one species are
particularly important and so you want a good representation in the training
set, then you would want to include as many as possible in the training set
during the seed part of the training.

Inversely, if a system has high representation of a species well-described
by a simple 2+3 body kernel, you may want it to be less well represented
in the seeded training set.

By specifying the pre_train_atoms_per_element, you can limit the number of
atoms of a given species which are added in. You can also limit the number
of atoms which are added from a given seed frame.

"""
import json as json
import logging
import numpy as np
import time
import warnings

from copy import deepcopy
from math import inf
from typing import List, Tuple, Union

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.mgp.mgp import MappedGaussianProcess
from flare.output import Output
from flare.predict import predict_on_atom, predict_on_atom_en, \
    predict_on_structure_par, predict_on_structure_par_en, \
    predict_on_structure_mgp
from flare.struc import Structure
from flare.utils.element_coder import element_to_Z, Z_to_element, NumpyEncoder
from flare.utils.learner import subset_of_frame_by_element, \
    is_std_in_bound_per_species, is_force_in_bound_per_species
from flare.mgp import MappedGaussianProcess
from flare.parameters import Parameters


class TrajectoryTrainer:

    def __init__(self, frames: List[Structure],
                 gp: Union[GaussianProcess, MappedGaussianProcess],
                 rel_std_tolerance: float = 4,
                 abs_std_tolerance: float = 1,
                 abs_force_tolerance: float = 0,
                 max_force_error: float = inf,
                 parallel: bool = False,
                 n_cpus: int = 1,
                 skip: int = 1,
                 validate_ratio: float = 0.0,
                 calculate_energy: bool = False,
                 output_name: str = 'gp_from_aimd',
                 print_as_xyz: bool = False,
                 pre_train_max_iter: int = 50,
                 max_atoms_from_frame: int = np.inf,
                 max_trains: int = np.inf,
                 min_atoms_per_train: int = 1,
                 shuffle_frames: bool = False,
                 verbose: str = "INFO",
                 pre_train_on_skips: int = -1,
                 pre_train_seed_frames: List[Structure] = None,
                 pre_train_seed_envs: List[Tuple[AtomicEnvironment,
                                                 'np.array']] = None,
                 pre_train_atoms_per_element: dict = None,
                 train_atoms_per_element: dict = None,
                 predict_atoms_per_element: dict = None,
                 train_checkpoint_interval: int = 1,
                 checkpoint_interval: int = 1,
                 atom_checkpoint_interval: int = 100,
                 print_training_plan: bool = True,
                 model_format: str = 'json'):
        """
        Class which trains a GP off of an AIMD trajectory, and generates
        error statistics between the DFT and GP calls.

        There are a variety of options which can give you a finer control
        over the training process.

        :param frames: List of structures to evaluate / train GP on
        :param gp: Gaussian Process object
        :param rel_std_tolerance: Train if uncertainty is above this *
            noise variance hyperparameter
        :param abs_std_tolerance: Train if uncertainty is above this
        :param abs_force_tolerance: Add atom force error exceeds this
        :param max_force_error: Don't add atom if force error exceeds this
        :param parallel: Use parallel functions or not
        :param validate_ratio: Fraction of frames used for validation
        :param skip: Skip through frames
        :param calculate_energy: Use local energy kernel or not
        :param output_name: Write output of training to this file
        :param print_as_xyz: If True, print the configurations in xyz format
        :param max_atoms_from_frame: Largest # of atoms added from one frame
        :param min_atoms_per_train: Only train when this many atoms have been
            added
        :param max_trains: Stop training GP after this many calls to train
        :param n_cpus: Number of CPUs to parallelize over for parallelization
                over atoms
        :param shuffle_frames: Randomize order of frames for better training
        :param verbose: same as logging level, "WARNING", "INFO", "DEBUG"
        :param pre_train_on_skips: Train model on every n frames before running
        :param pre_train_seed_frames: Frames to train on before running
        :param pre_train_seed_envs: Environments to train on before running
        :param pre_train_atoms_per_element: Max # of environments to add from
            each species in the seed pre-training steps
        :param train_atoms_per_element: Max # of environments to add from
            each species in the training steps
        :param predict_atoms_per_element: Choose a random subset of N random
            atoms from each specified element to predict on. For instance,
            {"H":5} will only predict the forces and uncertainties
            associated with 5 Hydrogen atoms per frame. Elements not
            specified will be predicted as normal. This is useful for
            systems where you are most interested in a subset of elements.
            This will result in a faster but less exhaustive learning process.
        :param checkpoint_interval: Will be deprecated. Same as
                            train_checkpoint_interval
        :param train_checkpoint_interval: How often to write model after
                        trainings
        :param atom_checkpoint_interval: How often to write model after atoms are
            added (since atoms may be added without training)
        :param model_format: Format to write GP model to
        :param print_training_plan: Write which atoms in which frames that
            triggered uncertainty or force conditions, so that training can
            be 'fast-forwarded' later. Also useful for gauging MGP results and
            then applying the atoms with high uncertainty and error to a GP.
        """

        # Set up parameters
        self.frames = frames
        if shuffle_frames:
            np.random.shuffle(frames)
            if print_training_plan:
                warnings.warn("Frames are shuffled so training plan will not"
                              " map onto the structures used; Try to "
                              "shuffle the frames outside of the GPFA module "
                              "for now.")

        # GP Training and Execution parameters
        self.gp = gp
        # Check to see if GP is MGP for later flagging
        self.mgp = isinstance(gp, MappedGaussianProcess)
        self.rel_std_tolerance = rel_std_tolerance
        self.abs_std_tolerance = abs_std_tolerance
        self.abs_force_tolerance = abs_force_tolerance
        self.max_force_error = max_force_error
        self.max_trains = max_trains
        self.max_atoms_from_frame = max_atoms_from_frame
        self.min_atoms_per_train = min_atoms_per_train
        self.predict_atoms_per_element = predict_atoms_per_element
        self.train_count = 0
        self.calculate_energy = calculate_energy
        self.n_cpus = n_cpus

        if parallel is True:
            warnings.warn("Parallel flag will be deprecated;"
                          "we will instead use n_cpu alone.",
                          DeprecationWarning)

        # Set prediction function based on if forces or energies are
        # desired, and parallelization accordingly
        if not self.mgp:
            if calculate_energy:
                self.pred_func = predict_on_structure_par_en
            else:
                self.pred_func = predict_on_structure_par

        elif self.mgp:
            self.pred_func = predict_on_structure_mgp

        # Parameters for negotiating with the training frames

        # To later be filled in using the time library
        self.start_time = None

        self.skip = skip
        assert (isinstance(skip, int) and skip >= 1), "Skip needs to be a " \
                                                      "positive integer."
        self.validate_ratio = validate_ratio
        assert (0 <= validate_ratio <= 1), \
            "validate_ratio needs to be [0,1]"

        # Set up for pretraining
        self.pre_train_max_iter = pre_train_max_iter
        self.pre_train_on_skips = pre_train_on_skips
        self.seed_envs = [] if pre_train_seed_envs is None else \
            pre_train_seed_envs
        self.seed_frames = [] if pre_train_seed_frames is None \
            else pre_train_seed_frames

        self.pre_train_env_per_species = {} if pre_train_atoms_per_element \
            is None else pre_train_atoms_per_element
        self.train_env_per_species = {} if train_atoms_per_element \
            is None else train_atoms_per_element

        # Convert to Coded Species
        if self.pre_train_env_per_species:
            pre_train_species = list(self.pre_train_env_per_species.keys())
            for key in pre_train_species:
                self.pre_train_env_per_species[element_to_Z(key)] = \
                    self.pre_train_env_per_species[key]

        # Output parameters
        self.output = Output(output_name, verbose,
                             print_as_xyz=print_as_xyz, always_flush=True)
        self.logger_name = self.output.basename+'log'
        self.train_checkpoint_interval = train_checkpoint_interval or \
            checkpoint_interval
        self.atom_checkpoint_interval = atom_checkpoint_interval

        self.model_format = model_format
        self.output_name = output_name
        self.print_training_plan = print_training_plan

        # Defining variables to be used later
        self.curr_step = 0
        self.train_count = 0
        self.start_time = time.time()

    def pre_run(self):
        """
        Various tasks to set up the AIMD training before commencing
        the run through the AIMD trajectory.
        1. Print the output.
        2. Pre-train the GP with the seed frames and
        environments. If no seed frames or environments and the GP has no
        training set, then seed with at least one atom from each
        """

        if self.mgp:
            raise NotImplementedError("Pre-running not"
                                      "yet configured for MGP")
        self.output.write_header(str(self.gp),
                                 dt=0,
                                 Nsteps=len(self.frames),
                                 structure=None,
                                 std_tolerance=(self.rel_std_tolerance,
                                                self.abs_std_tolerance),
                                 optional={
                                     'GP Statistics':
                                         json.dumps(
                                             self.gp.training_statistics),
                                     'GP Name': self.gp.name,
                                     'GP Write Name':
            self.output_name + "_model." + self.model_format})

        self.start_time = time.time()
        logger = logging.getLogger(self.logger_name)
        logger.debug("Now beginning pre-run activity.")
        # If seed environments were passed in, add them to the GP.

        for point in self.seed_envs:
            self.gp.add_one_env(point[0], point[1], train=False)

        # No training set ("blank slate" run) and no seeds specified:
        # Take one of each atom species in the first frame
        # so all atomic species are represented in the first step.
        # Otherwise use the seed frames passed in by user.

        # Remove frames used as seed from later part of training
        if self.pre_train_on_skips > 0:
            self.seed_frames = []
            newframes = []
            for i in range(len(self.frames)):
                if (i % self.pre_train_on_skips) == 0:
                    self.seed_frames += [self.frames[i]]
                else:
                    newframes += [self.frames[i]]
            self.frames = newframes

        # If the GP is empty, use the first frame as a seed frame.
        elif len(self.gp.training_data) == 0 and len(self.seed_frames) == 0:
            self.seed_frames = [self.frames[0]]
            self.frames = self.frames[1:]

        atom_count = 0
        for frame in self.seed_frames:
            train_atoms = []
            for species_i in set(frame.coded_species):
                # Get a randomized set of atoms of species i from the frame
                # So that it is not always the lowest-indexed atoms chosen
                atoms_of_specie = frame.indices_of_specie(species_i)
                np.random.shuffle(atoms_of_specie)
                n_at = len(atoms_of_specie)
                # Determine how many to add based on user defined cutoffs
                n_to_add = min(n_at, self.pre_train_env_per_species.get(
                    species_i, inf), self.max_atoms_from_frame)

                for atom in atoms_of_specie[:n_to_add]:
                    train_atoms.append(atom)
                    atom_count += 1

            self.update_gp_and_print(frame=frame,
                                     train_atoms=train_atoms,
                                     uncertainties=[], train=False)

        logger = logging.getLogger(self.logger_name)
        if atom_count > 0:
            logger.info(f"Added {atom_count} atoms to "
                        f"pretrain.\n"
                        f"Pre-run GP Statistics: "
                        f"{json.dumps(self.gp.training_statistics)} ")

        if (self.seed_envs or atom_count or self.seed_frames) and \
                (self.pre_train_max_iter or self.max_trains):
            logger.debug("Now commencing pre-run training of GP (which has "
                         "non-empty training set)")
            time0 = time.time()
            self.train_gp(max_iter=self.pre_train_max_iter)
            logger.debug(f"Done train_gp {time.time()-time0}")
        else:
            logger.debug("Now commencing pre-run set up of GP (which has "
                         "non-empty training set)")
            time0 = time.time()
            self.gp.check_L_alpha()
            logger.debug(f"Done check_L_alpha {time.time()-time0}")

        if self.model_format and not self.mgp:
            self.gp.write_model(f'{self.output_name}_prerun',
                                self.model_format)

    def run(self):
        """
        Loop through frames and record the error between
        the GP predictions and the ground-truth forces. Train the GP and update
        the training set upon the triggering of the uncertainty or force error
        threshold.

        :return: None
        """

        # Perform pre-run, in which seed trames are used.
        logger = logging.getLogger(self.logger_name)
        logger.debug("Commencing run with pre-run...")
        if not self.mgp:
            self.pre_run()

        # Past this frame, stop adding atoms to the training set
        #  (used for validation of model)
        train_frame = int(len(self.frames[::self.skip])
                          * (1 - self.validate_ratio))

        # Loop through trajectory.
        cur_atoms_added_train = 0  # Track atoms added for training
        cur_atoms_added_write = 0  # Track atoms added for writing
        cur_trains_done_write = 0  # Track training done for writing

        # Keep track of which atoms trigger force / uncertainty condition
        training_plan = {}

        for i, cur_frame in enumerate(self.frames[::self.skip]):

            frame_start_time = time.time()
            logger.info(f"=====NOW ON FRAME {i}=====")

            # If no predict_atoms_per_element was specified, predict_atoms
            # will be equal to every atom in the frame.
            predict_atoms = subset_of_frame_by_element(
                cur_frame, self.predict_atoms_per_element)

            # Atoms which are skipped will have NaN as their force / std values
            local_energies = None

            # Three different predictions: Either MGP, GP with energy,
            # or GP without
            if self.mgp:
                pred_forces, pred_stds, local_energies = self.pred_func(
                    structure=cur_frame, mgp=self.gp, write_to_structure=False,
                    selective_atoms=predict_atoms, skipped_atom_value=np.nan,
                    energy=True)
            elif self.calculate_energy:
                pred_forces, pred_stds, local_energies = self.pred_func(
                    structure=cur_frame, gp=self.gp, n_cpus=self.n_cpus,
                    write_to_structure=False, selective_atoms=predict_atoms,
                    skipped_atom_value=np.nan)
            else:
                pred_forces, pred_stds = self.pred_func(
                    structure=cur_frame, gp=self.gp, n_cpus=self.n_cpus,
                    write_to_structure=False, selective_atoms=predict_atoms,
                    skipped_atom_value=np.nan)

            # Get Error
            dft_forces = cur_frame.forces
            dft_energy = cur_frame.energy
            error = np.abs(pred_forces - dft_forces)

            # Create dummy frame with the predicted forces written
            dummy_frame = deepcopy(cur_frame)
            dummy_frame.forces = pred_forces
            dummy_frame.stds = pred_stds

            self.output.write_gp_dft_comparison(
                curr_step=i, frame=dummy_frame, start_time=time.time(),
                dft_forces=dft_forces, dft_energy=dft_energy, error=error,
                local_energies=local_energies, KE=0)

            logger.debug(
                f'Single frame calculation time {time.time()-frame_start_time}')

            if i < train_frame:
                # Noise hyperparameter & relative std tolerance is not for mgp.
                if self.mgp:
                    noise = 0
                else:
                    noise = Parameters.get_noise(
                        self.gp.hyps_mask, self.gp.hyps, constraint=False)
                std_in_bound, std_train_atoms = is_std_in_bound_per_species(
                    rel_std_tolerance=self.rel_std_tolerance,
                    abs_std_tolerance=self.abs_std_tolerance,
                    noise=noise, structure=dummy_frame,
                    max_atoms_added=self.max_atoms_from_frame,
                    max_by_species=self.train_env_per_species)

                # Get max force error atoms
                force_in_bound, force_train_atoms = \
                    is_force_in_bound_per_species(
                        abs_force_tolerance=self.abs_force_tolerance,
                        predicted_forces=pred_forces,
                        label_forces=dft_forces,
                        structure=dummy_frame,
                        max_atoms_added=self.max_atoms_from_frame,
                        max_by_species=self.train_env_per_species,
                        max_force_error=self.max_force_error)

                if not std_in_bound or not force_in_bound:

                    # -1 is returned from the is_in_bound methods,
                    # so filter that out and the use sets to remove repeats
                    train_atoms = list(set(std_train_atoms).union(
                        force_train_atoms) - {-1})

                    training_plan[int(i)] = [int(a) for a in train_atoms]

                    # Compute mae and write to output;
                    # Add max uncertainty atoms to training set
                    self.update_gp_and_print(
                        cur_frame, train_atoms=train_atoms,
                        uncertainties=pred_stds[train_atoms],
                        train=False)
                    cur_atoms_added_train += len(train_atoms)
                    cur_atoms_added_write += len(train_atoms)
                    # Re-train if number of sampled atoms is high enough

                    if cur_atoms_added_train >= self.min_atoms_per_train or (
                            i + 1) == train_frame:
                        if self.train_count < self.max_trains:
                            self.train_gp()
                            cur_trains_done_write += 1
                        else:
                            self.gp.update_L_alpha()
                        cur_atoms_added_train = 0
                    else:
                        self.gp.update_L_alpha()

                    # Loop to decide of a model should be written this
                    # iteration
                    will_write = False

                    if self.train_checkpoint_interval and \
                            cur_trains_done_write and \
                            self.train_checkpoint_interval \
                        <= cur_trains_done_write:
                        will_write = True
                        cur_trains_done_write = 0

                    if self.atom_checkpoint_interval \
                            and cur_atoms_added_write \
                            and self.atom_checkpoint_interval \
                        <= cur_atoms_added_write:
                        will_write = True
                        cur_atoms_added_write = 0

                    if self.model_format and will_write:
                        self.gp.write_model(f'{self.output_name}_checkpt',
                                            self.model_format)

                if (i + 1) == train_frame and not self.mgp:
                    self.gp.check_L_alpha()

        #Print training statistics for GP model used
        conclusion_strings = []
        conclusion_strings.append('Final GP statistics:'
                                  + json.dumps(self.gp.training_statistics)
                                  )
        self.output.conclude_run(conclusion_strings)

        if self.print_training_plan:
            with open(f'{self.output_name}_training_plan.json', 'w') as f:
                f.write(json.dumps(training_plan, cls=NumpyEncoder))

        if self.model_format and not self.mgp:
            self.gp.write_model(f'{self.output_name}_model',
                                self.model_format)

    def update_gp_and_print(self, frame: Structure, train_atoms: List[int],
                            uncertainties: List[int] = None,
                            train: bool = True):
        """
        Update the internal GP model training set with a list of training
        atoms indexing atoms within the frame. If train is True, re-train
        the GP by optimizing hyperparameters.
        :param frame: Structure to train on
        :param train_atoms: Index atoms to train on
        :param uncertainties: Uncertainties to print, pass in [] to silence
        :param train: Train or not
        :return: None
        """

        # Group added atoms by species for easier output
        added_species = [Z_to_element(frame.coded_species[at]) for at in
                         train_atoms]
        added_atoms = {spec: [] for spec in set(added_species)}

        for atom, spec in zip(train_atoms, added_species):
            added_atoms[spec].append(atom)

        logger = logging.getLogger(self.logger_name)
        logger.info('Adding atom(s) '
                    f'{json.dumps(added_atoms,cls=NumpyEncoder)}'
                    ' to the training set.')

        if uncertainties is None or len(uncertainties) != 0:
            uncertainties = frame.stds[train_atoms]

        if len(uncertainties) != 0:
            logger.info(f'Uncertainties: '
                        f'{uncertainties}.')

        # update gp model; handling differently if it's an MGP
        if not self.mgp:
            self.gp.update_db(frame, frame.forces, custom_range=train_atoms)

            if train:
                self.train_gp()

        else:
            logger.warning("Warning: Adding data to an MGP is not yet "
                           "supported.")

    def train_gp(self, max_iter: int = None):
        """
        Train the Gaussian process and write the results to the output file.

        :param max_iter: Maximum iterations associated with this training run,
            overriding the Gaussian Process's internally set maxiter.
        :type max_iter: int
        """
        logger = logging.getLogger(self.logger_name)

        if self.mgp:
            logger.debug("Training skipped because of MGP")
            return

        logger.debug('Train GP')

        logger_train = self.output.basename+'hyps'

        # TODO: Improve flexibility in GP training to make this next step
        # unnecessary, so maxiter can be passed as an argument

        # Don't train if maxiter == 0
        if max_iter == 0:
            self.gp.check_L_alpha()
        elif max_iter is not None:
            temp_maxiter = self.gp.maxiter
            self.gp.maxiter = max_iter
            self.gp.train(logger_name=logger_train)
            self.gp.maxiter = temp_maxiter
        else:
            self.gp.train(logger_name=logger_train)

        hyps, labels = Parameters.get_hyps(self.gp.hyps_mask, self.gp.hyps, constraint=False,
                        label=True)
        if labels is None:
            labels = self.gp.hyp_labels
        self.output.write_hyps(labels, hyps,
                               self.start_time,
                               self.gp.likelihood,
                               self.gp.likelihood_gradient,
                               hyps_mask=self.gp.hyps_mask)
        self.train_count += 1


def parse_trajectory_trainer_output(file: str, return_gp_data: bool = False,
                                    compute_errors: bool = True) \
        -> Union[List[dict], Tuple[List[dict], dict]]:
    """
    Reads output of a TrajectoryTrainer run by frame. return_gp_data returns
    data about GP model growth useful for visualizing progress of model
    training.

    :param file: filename of output
    :param return_gp_data: flag for returning extra GP data
    :param compute_errors: Compute deviation from GP and DFT forces.
    :return: List of dictionaries with keys 'species', 'positions',
        'gp_forces', 'dft_forces', 'gp_stds', 'added_atoms', and
        'maes_by_species', optionally, gp_data dictionary
    """

    with open(file, 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)

    # Get indexes where frames begin, and include the index of the final line
    frame_indexes = [i for i in range(num_lines) if '-Frame:' in
                     lines[i]] + [num_lines]

    frames = []

    # Central parsing loop
    for n in range(len(frame_indexes) - 1):
        # Start at +2 to skip frame marker and header of table of data
        # Set up values for current frame which will be populated

        frame_atoms = []
        frame_positions = []
        gp_forces = []
        dft_forces = []
        stds = []
        added_atoms = {}
        frame_species_maes = {}

        # i loops through individual atom's info
        for i in range(frame_indexes[n] + 2, frame_indexes[n + 1]):

            # Lines with data will be long; stop when at end of atom data
            if len(lines[i]) > 10:
                split = lines[i].split()

                frame_atoms.append(split[0])

                frame_positions.append([float(split[1]), float(split[2]),
                                        float(split[3])])
                gp_forces.append([float(split[4]), float(split[5]),
                                  float(split[6])])
                stds.append(
                    [float(split[7]), float(split[8]), float(split[9])])

                dft_forces.append([float(split[10]), float(split[11]),
                                   float(split[12])])

            # Terminate at blank line between results
            else:
                break
        # Loop through information in frame after Data
        for i in range(frame_indexes[n] + len(frame_positions) + 2,
                       frame_indexes[n + 1]):

            if 'Adding atom(s)' in lines[i]:
                # Splitting to target the 'added atoms' substring
                split_line = lines[i][15:-21]
                added_atoms = json.loads(split_line.strip())

            if 'type ' in lines[i]:
                cur_line = lines[i].split()
                frame_species_maes[cur_line[1]] = float(cur_line[3])

        cur_frame_stats = {'species': frame_atoms,
                           'positions': np.array(frame_positions),
                           'gp_forces': np.array(gp_forces),
                           'dft_forces': np.array(dft_forces),
                           'gp_stds': np.array(stds),
                           'added_atoms': added_atoms,
                           'maes_by_species': frame_species_maes}

        if compute_errors:
            cur_frame_stats['force_errors'] = np.array(gp_forces) \
                                             - np.array(dft_forces)
        frames.append(cur_frame_stats)

    if not return_gp_data:
        return frames

    # Compute information about GP training
    # to study GP growth and performance over trajectory

    gp_stats_line = [line for line in lines[:30] if 'GP Statistics' in
                     line and 'Pre-run' not in line][0][15:].strip()

    initial_gp_statistics = json.loads(gp_stats_line)

    # Get pre_run statistics (if pre-run was done):
    pre_run_gp_statistics = None
    pre_run_gp_stats_line = [line for line in lines if 'Pre-run GP' in line]
    if pre_run_gp_stats_line:
        pre_run_gp_statistics = json.loads(pre_run_gp_stats_line[0][
                                           22:].strip())

    # Compute cumulative GP size
    cumulative_gp_size = [int(initial_gp_statistics['N'])]

    if pre_run_gp_stats_line:
        cumulative_gp_size.append(int(pre_run_gp_statistics['N']))

    running_total = cumulative_gp_size[-1]

    for frame in frames:

        added_atom_dict = frame['added_atoms']
        for val in added_atom_dict.values():
            running_total += len(val)
        cumulative_gp_size.append(running_total)

    # Compute MAEs for each element over time
    all_species = set()
    for frame in frames:
        all_species = all_species.union(set(frame['species']))

    all_species = list(all_species)
    mae_by_elt = {elt: [] for elt in all_species}

    for frame in frames:
        for elt in all_species:
            cur_mae = frame['maes_by_species'].get(elt, np.nan)
            mae_by_elt[elt].append(cur_mae)

    gp_data = {'init_stats': initial_gp_statistics,
               'pre_train_stats': pre_run_gp_statistics,
               'cumulative_gp_size': cumulative_gp_size,
               'mae_by_elt': mae_by_elt}

    return frames, gp_data
