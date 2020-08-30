"""
Tool to enable the development of a GP model based on an AIMD
trajectory with many customizable options for fine control of training.
Contains methods to transfer the model to an OTF run or MD engine run.
"""
import json as json
import logging
import numpy as np
import time
import random as random
import warnings

from copy import deepcopy
from math import inf
from typing import List, Tuple, Union, Iterable, Dict

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.mgp.mgp import MappedGaussianProcess
from flare.output import Output
from flare.predict import (
    predict_on_atom,
    predict_on_atom_en,
    predict_on_structure_par,
    predict_on_structure_par_en,
    predict_on_structure_mgp,
)
from flare.struc import Structure
from flare.utils.element_coder import element_to_Z, Z_to_element, NumpyEncoder
from flare.utils.learner import (
    subset_of_frame_by_element,
    is_std_in_bound_per_species,
    is_force_in_bound_per_species,
)
from flare.mgp import MappedGaussianProcess
from flare.parameters import Parameters
from flare.output import Output


class LearningProtocol(object):
    def __init__(
        self,
        model: Union[GaussianProcess, MappedGaussianProcess],
        n_cpus: int = 0,
        protocol_name: str = "learning",
    ):

        self.model = model
        self.output = Output(basename=protocol_name, always_flush=True)

    def train_model(self, **train_kwargs):

        if isinstance(self.model, GaussianProcess):
            self.model.train(train_kwargs)

        elif isinstance(self.model, MappedGaussianProcess):
            raise NotImplementedError("MGP training not defined")

        raise NotImplementedError("Must be implemented in child class")

    def augment_training_set(
        self,
        structure: Structure,
        selective_atoms: List[int] = list(),
        forces: Iterable[float] = None,
        energy: float = None,
    ):

        if forces is None:
            forces = structure.forces
        if energy is None and forces is None:
            energy = structure.energy

        if isinstance(self.model, GaussianProcess):

            self.model.update_db(
                structure=structure,
                selective_atoms=selective_atoms,
                forces=forces,
                energy=energy,
            )

        raise NotImplementedError("Must be implemented in child class")

    def get_next_structure(self):
        raise NotImplementedError("Must be implemented in child class")

    def check_prediction(self):
        raise NotImplementedError("Must be implemented in child class")

    def run_active_learning(self, init_frames: List[Structure]):
        """

        :param init_frames:
        :return:
        """
        raise NotImplementedError("Must be implemented in child class")

    def pre_run(self):

        if isinstance(self.model, GaussianProcess):
            self.output.write_header(
                str(self.gp),
                dt=0,
                Nsteps=len(self.frames),
                structure=None,
                optional={
                    "GP Statistics": json.dumps(self.gp.training_statistics),
                    "GP Name": self.gp.name,
                    "GP Write Name": self.output_name + "_model." + self.model_format,
                },
            )

    def run_passive_learning(
        self,
        frames: List[Structure] = list(),
        envs: List[AtomicEnvironment] = list(),
        max_atoms_from_frame: int = inf,
        shuffle_frames: bool = False,
        max_atoms_added: int = 1000,
        max_from_frame_by_species: Dict[str, int] = dict(),
    ):

        if not isinstance(self.model, GaussianProcess):
            raise NotImplementedError("Passive learning notyet configured for MGP")

        self.start_time = time.time()
        logger = logging.getLogger(self.logger_name)
        logger.debug("Now beginning passive learning.")

        # If seed environments were passed in, add them to the GP.

        for env in envs:
            self.model.add_one_env(env[0], env[1], train=False)
            logger.info("Added {str(env)} to dataset.")

        if shuffle_frames:
            frames = random.sample(frames, len(frames))

        n_atoms_added = 0

        for frame in frames:
            train_atoms = []
            # Get train atoms by species
            for species_i in set(frame.coded_species):
                # Get a randomized set of atoms of species i from the frame
                # So that it is not always the lowest-indexed atoms chosen
                atoms_of_specie = frame.indices_of_specie(species_i)
                n_spec = len(atoms_of_specie)
                # Determine how many to add based on user defined cutoffs
                n_to_add = min(
                    n_spec,
                    max_from_frame_by_species.get(species_i, inf),
                    max_atoms_from_frame,
                )

                train_atoms += random.sample(atoms_of_specie, n_spec)
                n_atoms_added += n_spec

            self.model.update_db(struc=frame, custom_range=train_atoms)
            if n_atoms_added >= max_atoms_added:
                break

        if n_atoms_added:
            logger.info(
                f"Added {n_atoms_added} atoms to "
                "pretrain.\n"
                "Pre-run GP Statistics: "
                f"{json.dumps(self.model.training_statistics)} "
            )

        if (envs or n_atoms_added or frames) and (
            self.pre_train_max_iter or self.max_trains
        ):
            logger.debug(
                "Now commencing pre-run training of GP (which has "
                "non-empty training set)"
            )
            time0 = time.time()
            self.train_gp(max_iter=self.pre_train_max_iter)
            logger.debug(f"Done train_gp {time.time()-time0}")
        else:
            logger.debug(
                "Now commencing pre-run set up of GP (which has non-empty training set)"
            )
            time0 = time.time()
            self.gp.check_L_alpha()
            logger.debug(f"Done check_L_alpha {time.time()-time0}")

        if self.model_format and not self.mgp:
            self.gp.write_model(f"{self.output_name}_prerun", self.model_format)


class TrainFromAIMD(LearningProtocol):
    def __init__(self):
        pass

    def train_model(self):
        pass


class LearningProtocol:
    def __init__(
        self,
        gp: Union[GaussianProcess, MappedGaussianProcess],
        active_frames: List[Structure] = None,
        passive_frames: List[Structure] = None,
        passive_envs: List[Tuple[AtomicEnvironment, "np.array"]] = None,
        active_rel_var_tol: float = 4,
        active_abs_var_tol: float = 1,
        active_abs_error_tol: float = 0,
        active_error_tol_cutoff: float = inf,
        active_max_trains: int = np.inf,
        active_max_element_from_frame: dict = None,
        checkpoint_interval_train: int = 1,
        checkpoint_interval_atom: int = 100,
        predict_atoms_per_element: dict = None,
        max_atoms_from_frame: int = np.inf,
        min_atoms_added_per_train: int = 1,
        max_model_size: int = np.inf,
        passive_on_active_skips: int = -1,
        passive_train_max_iter: int = 50,
        passive_atoms_per_element: dict = None,
        active_skip: int = 1,
        shuffle_active_frames: bool = False,
        n_cpus: int = 1,
        validate_ratio: float = 0.0,
        calculate_energy: bool = False,
        output_name: str = "gp_from_aimd",
        print_as_xyz: bool = False,
        verbose: str = "INFO",
        written_model_format: str = "json",
    ):
        """
        Class which trains a GP off of an AIMD trajectory, and generates
        error statistics between the DFT and GP calls.
        All arguments are divided between 'passive' learning and 'active'
        learning. By default, when run is called, a 'passive' learning run
        is called which either adds all 'seed' environments to the model,
        or a randomized subset of atoms from the frames. If no arguments are
        specified, the very first frame of the active learning
        frames will be used.
        "Passive" learning will add data based on random selection of atoms
        from a given ab-initio frame.
        "Active" learning will add data to the dataset based on the
        performance of the GP itself: the force error and the GP's internal
        uncertainty estimate.
        There are a widevariety of options which can give you a finer
        control over the training process.
        :param active_frames: List of structures to evaluate / train GP on
        :param gp: Gaussian Process object
        :param active_rel_var_tol: Train if uncertainty is above this *
            noise variance hyperparameter
        :param active_abs_var_tol: Train if uncertainty is above this
        :param active_abs_error_tol: Add atom force error exceeds this
        :param active_error_tol_cutoff: Don't add atom if force error exceeds this
        :param validate_ratio: Fraction of frames used for validation
        :param active_skip: Skip through frames
        :param calculate_energy: Use local energy kernel or not
        :param output_name: Write output of training to this file
        :param print_as_xyz: If True, print the configurations in xyz format
        :param max_atoms_from_frame: Largest # of atoms added from one frame
        :param min_atoms_added_per_train: Only train when this many atoms have been
            added
        :param active_max_trains: Stop training GP after this many calls to train
        :param n_cpus: Number of CPUs to parallelize over for parallelization
                over atoms
        :param shuffle_active_frames: Randomize order of frames for better training
        :param verbose: same as logging level, "WARNING", "INFO", "DEBUG"
        :param passive_on_active_skips: Train model on every n frames before running
        :param passive_frames: Frames to train on before running
        :param passive_envs: Environments to train on before running
        :param passive_atoms_per_element: Max # of environments to add from
            each species in the seed pre-training steps
        :param active_max_element_from_frame: Max # of environments to add from
            each species in the training steps
        :param predict_atoms_per_element: Choose a random subset of N random
            atoms from each specified element to predict on. For instance,
            {"H":5} will only predict the forces and uncertainties
            associated with 5 Hydrogen atoms per frame. Elements not
            specified will be predicted as normal. This is useful for
            systems where you are most interested in a subset of elements.
            This will result in a faster but less exhaustive learning process.
        :param checkpoint_interval_train: How often to write model after
                        trainings
        :param checkpoint_interval_atom: How often to write model after atoms are
            added (since atoms may be added without training)
        :param written_model_format: Format to write GP model to
        """

        # GP Training and Execution parameters
        self.gp = gp
        # Check to see if GP is MGP for later flagging
        self.mgp = isinstance(gp, MappedGaussianProcess)

        self.rel_std_tolerance = active_rel_var_tol
        self.abs_std_tolerance = active_abs_var_tol
        self.abs_force_tolerance = active_abs_error_tol
        self.max_force_error = active_error_tol_cutoff
        self.max_trains = active_max_trains
        self.max_atoms_from_frame = max_atoms_from_frame
        self.min_atoms_per_train = min_atoms_added_per_train
        self.max_model_size = max_model_size

        # Set prediction function based on if forces or energies are
        # desired, and parallelization accordingly
        if not self.mgp:
            if calculate_energy:
                self.pred_func = predict_on_structure_par_en
            else:
                self.pred_func = predict_on_structure_par

        elif self.mgp:
            self.pred_func = predict_on_structure_mgp

        self.start_time = time.time()

        self.train_count = 0
        self.calculate_energy = calculate_energy
        self.n_cpus = n_cpus

        # Output parameters
        self.output = Output(
            output_name, verbose, print_as_xyz=print_as_xyz, always_flush=True
        )
        self.logger_name = self.output.basename + "log"
        self.train_checkpoint_interval = checkpoint_interval_train
        self.atom_checkpoint_interval = checkpoint_interval_atom

        self.model_format = written_model_format
        self.output_name = output_name

        # gpfa only function

        self.predict_atoms_per_element = predict_atoms_per_element

        # Set up parameters
        self.frames = active_frames
        if shuffle_active_frames:
            np.random.shuffle(active_frames)

        # Parameters for negotiating with the training active_frames
        self.skip = active_skip
        assert (
            isinstance(active_skip, int) and active_skip >= 1
        ), "Skip needs to be a  positive integer."
        self.validate_ratio = validate_ratio
        assert 0 <= validate_ratio <= 1, "validate_ratio needs to be [0,1]"

        # Set up for pretraining
        self.pre_train_max_iter = passive_train_max_iter
        self.pre_train_on_skips = passive_on_active_skips
        self.seed_envs = [] if passive_envs is None else passive_envs
        self.seed_frames = [] if passive_frames is None else passive_frames

        self.pre_train_env_per_species = (
            {} if passive_atoms_per_element is None else passive_atoms_per_element
        )
        self.train_env_per_species = (
            {}
            if active_max_element_from_frame is None
            else active_max_element_from_frame
        )

        # Convert to Coded Species
        if self.pre_train_env_per_species:
            pre_train_species = list(self.pre_train_env_per_species.keys())
            for key in pre_train_species:
                self.pre_train_env_per_species[
                    element_to_Z(key)
                ] = self.pre_train_env_per_species[key]

        # Defining variables to be used later
        self.curr_step = 0
        self.train_count = 0
        self.start_time = time.time()

    def get_next_env(self):
        self.curr_env_index += 1
        if self.curr_env_index < len(self.seed_envs):
            return self.seed_envs[self.curr_env_index]
        return None

    def get_next_passive_frame(self):
        self.curr_passive_frame_index += 1
        if self.curr_passive_frame_index < len(self.seed_frames):
            return self.seed_frames[self.curr_passive_frame_index]
        return None

    def preparation_for_passive_run(self):
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

    def preparation_for_active_run(self):
        raise NotImplementedError("need to be implemented in child class")

    def get_next_active_frame(self):
        raise NotImplementedError("need to be implemented in child class")

    def decide_to_update_db(self):
        raise NotImplementedError("need to be implemented in child class")

    def decide_to_checkLalpha(self):
        raise NotImplementedError("need to be implemented in child class")

    def passive_run(self):
        """
        Various tasks to set up the AIMD training before commencing
        the run through the AIMD trajectory.
        1. Print the output.
        2. Pre-train the GP with the seed frames and
        environments. If no seed frames or environments and the GP has no
        training set, then seed with at least one atom from each
        """

        if self.mgp:
            raise NotImplementedError("Pre-running notyet configured for MGP")
        self.output.write_header(
            str(self.gp),
            dt=0,
            Nsteps=len(self.frames),
            structure=None,
            std_tolerance=(self.rel_std_tolerance, self.abs_std_tolerance),
            optional={
                "GP Statistics": json.dumps(self.gp.training_statistics),
                "GP Name": self.gp.name,
                "GP Write Name": self.output_name + "_model." + self.model_format,
            },
        )

        self.start_time = time.time()
        logger = logging.getLogger(self.logger_name)
        logger.debug("Now beginning pre-run activity.")

        # If seed environments were passed in, add them to the GP.

        self.preparation_for_passive_run()

        self.curr_env_index = -1
        curr_env = self.get_next_env()
        while curr_env is not None:
            self.gp.add_one_env(curr_env[0], curr_env[1], train=False)
            curr_env = self.get_next_env()

        # No training set ("blank slate" run) and no seeds specified:
        # Take one of each atom species in the first frame
        # so all atomic species are represented in the first step.
        # Otherwise use the seed frames passed in by user.

        self.passive_atom_count = 0
        self.curr_passive_frame_index = -1
        frame = self.get_next_passive_frame()
        while frame is not None:

            train_atoms = []
            for species_i in set(frame.coded_species):
                # Get a randomized set of atoms of species i from the frame
                # So that it is not always the lowest-indexed atoms chosen
                atoms_of_specie = frame.indices_of_specie(species_i)
                np.random.shuffle(atoms_of_specie)
                n_at = len(atoms_of_specie)
                # Determine how many to add based on user defined cutoffs
                n_to_add = min(
                    n_at,
                    self.pre_train_env_per_species.get(species_i, inf),
                    self.max_atoms_from_frame,
                )

                for atom in atoms_of_specie[:n_to_add]:
                    train_atoms.append(atom)
                    self.passive_atom_count += 1

            self.update_gp_and_print(
                frame=frame, train_atoms=train_atoms, uncertainties=[], train=False
            )
            frame = self.get_next_passive_frame()

        logger = logging.getLogger(self.logger_name)
        if self.passive_atom_count > 0:
            logger.info(
                f"Added {self.passive_atom_count} atoms to "
                "pretrain.\n"
                "Pre-run GP Statistics: "
                f"{json.dumps(self.gp.training_statistics)} "
            )

        if (self.seed_envs or self.passive_atom_count or self.seed_frames) and (
            self.pre_train_max_iter or self.max_trains
        ):
            logger.debug(
                "Now commencing pre-run training of GP (which has "
                "non-empty training set)"
            )
            time0 = time.time()
            self.train_gp(max_iter=self.pre_train_max_iter)
            logger.debug(f"Done train_gp {time.time()-time0}")
        else:
            logger.debug(
                "Now commencing pre-run set up of GP (which has non-empty training set)"
            )
            time0 = time.time()
            self.gp.check_L_alpha()
            logger.debug(f"Done check_L_alpha {time.time()-time0}")

        if self.model_format and not self.mgp:
            self.gp.write_model(f"{self.output_name}_prerun", self.model_format)

    def active_run(self):
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
            if len(self.gp) == 0:
                logger.warning(
                    "You are attempting to train a model with no "
                    "data in your Gausian Process; it is "
                    "recommended that you begin with "
                    "a passive training process."
                )

        self.preparation_for_active_run()

        # Loop through trajectory.
        self.cur_atoms_added_train = 0  # Track atoms added for training
        cur_atoms_added_write = 0  # Track atoms added for writing
        cur_trains_done_write = 0  # Track training done for writing

        self.curr_active_frame_index = -1
        cur_frame = self.get_next_active_frame()
        while cur_frame is not None:

            frame_start_time = time.time()
            logger.info(f"=====NOW ON FRAME {self.curr_active_frame_index}=====")

            # If no predict_atoms_per_element was specified, predict_atoms
            # will be equal to every atom in the frame.
            predict_atoms = subset_of_frame_by_element(
                cur_frame, self.predict_atoms_per_element
            )

            # Atoms which are skipped will have NaN as their force / std values
            local_energies = None

            # Three different predictions: Either MGP, GP with energy,
            # or GP without
            if self.mgp:
                pred_forces, pred_stds, local_energies = self.pred_func(
                    structure=cur_frame,
                    mgp=self.gp,
                    write_to_structure=False,
                    selective_atoms=predict_atoms,
                    skipped_atom_value=np.nan,
                    energy=True,
                )
            elif self.calculate_energy:
                pred_forces, pred_stds, local_energies = self.pred_func(
                    structure=cur_frame,
                    gp=self.gp,
                    n_cpus=self.n_cpus,
                    write_to_structure=False,
                    selective_atoms=predict_atoms,
                    skipped_atom_value=np.nan,
                )
            else:
                pred_forces, pred_stds = self.pred_func(
                    structure=cur_frame,
                    gp=self.gp,
                    n_cpus=self.n_cpus,
                    write_to_structure=False,
                    selective_atoms=predict_atoms,
                    skipped_atom_value=np.nan,
                )

            # Get Error
            dft_forces = cur_frame.forces
            dft_energy = cur_frame.energy
            error = np.abs(pred_forces - dft_forces)

            # Create dummy frame with the predicted forces written
            dummy_frame = deepcopy(cur_frame)
            dummy_frame.forces = pred_forces
            dummy_frame.stds = pred_stds

            self.output.write_gp_dft_comparison(
                curr_step=self.curr_active_frame_index,
                frame=dummy_frame,
                start_time=time.time(),
                dft_forces=dft_forces,
                dft_energy=dft_energy,
                error=error,
                local_energies=local_energies,
                KE=0,
            )

            logger.debug(
                f"Single frame calculation time {time.time()-frame_start_time}"
            )

            if self.decide_to_update_db():

                # Noise hyperparameter & relative std tolerance is not for mgp.
                if self.mgp:
                    noise = 0
                else:
                    noise = Parameters.get_noise(
                        self.gp.hyps_mask, self.gp.hyps, constraint=False
                    )

                std_in_bound, std_train_atoms = is_std_in_bound_per_species(
                    rel_std_tolerance=self.rel_std_tolerance,
                    abs_std_tolerance=self.abs_std_tolerance,
                    noise=noise,
                    structure=dummy_frame,
                    max_atoms_added=self.max_atoms_from_frame,
                    max_by_species=self.train_env_per_species,
                )

                # Get max force error atoms
                force_in_bound, force_train_atoms = is_force_in_bound_per_species(
                    abs_force_tolerance=self.abs_force_tolerance,
                    predicted_forces=pred_forces,
                    label_forces=dft_forces,
                    structure=dummy_frame,
                    max_atoms_added=self.max_atoms_from_frame,
                    max_by_species=self.train_env_per_species,
                    max_force_error=self.max_force_error,
                )

                if not std_in_bound or not force_in_bound:

                    # -1 is returned from the is_in_bound methods,
                    # so filter that out and the use sets to remove repeats
                    train_atoms = list(
                        set(std_train_atoms).union(force_train_atoms) - {-1}
                    )

                    # Compute mae and write to output;
                    # Add max uncertainty atoms to training set
                    self.update_gp_and_print(
                        cur_frame,
                        train_atoms=train_atoms,
                        uncertainties=pred_stds[train_atoms],
                        train=False,
                    )
                    self.cur_atoms_added_train += len(train_atoms)
                    cur_atoms_added_write += len(train_atoms)
                    # Re-train if number of sampled atoms is high enough

                    if self.decide_to_train():
                        self.train_gp()
                        cur_trains_done_write += 1
                        self.cur_atoms_added_train = 0
                    else:
                        self.gp.update_L_alpha()
                        # self.cur_atoms_added_train = 0

                    # Loop to decide of a model should be written this
                    # iteration
                    will_write = False

                    if (
                        self.train_checkpoint_interval
                        and cur_trains_done_write
                        and self.train_checkpoint_interval <= cur_trains_done_write
                    ):
                        will_write = True
                        cur_trains_done_write = 0

                    if (
                        self.atom_checkpoint_interval
                        and cur_atoms_added_write
                        and self.atom_checkpoint_interval <= cur_atoms_added_write
                    ):
                        will_write = True
                        cur_atoms_added_write = 0

                    if self.model_format and will_write:
                        self.gp.write_model(
                            f"{self.output_name}_checkpt", self.model_format
                        )

                if self.decide_to_checkLalpha():
                    self.gp.check_L_alpha()

            cur_frame = self.get_next_active_frame()

        self.output.conclude_run()

        if self.model_format and not self.mgp:
            self.gp.write_model(f"{self.output_name}_model", self.model_format)

    def update_gp_and_print(
        self,
        frame: Structure,
        train_atoms: List[int],
        uncertainties: List[int] = None,
        train: bool = True,
    ):
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
        added_species = [Z_to_element(frame.coded_species[at]) for at in train_atoms]
        added_atoms = {spec: [] for spec in set(added_species)}

        for atom, spec in zip(train_atoms, added_species):
            added_atoms[spec].append(atom)

        logger = logging.getLogger(self.logger_name)
        logger.info(
            "Adding atom(s) "
            f"{json.dumps(added_atoms,cls=NumpyEncoder)}"
            " to the training set."
        )

        if uncertainties is None or len(uncertainties) != 0:
            uncertainties = frame.stds[train_atoms]

        if len(uncertainties) != 0:
            logger.info(f"Uncertainties: {uncertainties}.")

        # update gp model; handling differently if it's an MGP
        if not self.mgp:
            self.gp.update_db(frame, frame.forces, custom_range=train_atoms)

            if train:
                self.train_gp()

        else:
            raise NotImplementedError

    def train_gp(self, max_iter: int = None):
        """
        Train the Gaussian process and write the results to the output file.
        :param max_iter: Maximum iterations associated with this training run,
            overriding the Gaussian Process's internally set maxiter.
        :type max_iter: int
        """

        logger = logging.getLogger(self.logger_name)
        logger.debug("Train GP")

        logger_train = self.output.basename + "hyps"

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

        hyps, labels = Parameters.get_hyps(
            self.gp.hyps_mask, self.gp.hyps, constraint=False, label=True
        )
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
        self.train_count += 1


def parse_trajectory_trainer_output(
    file: str, return_gp_data: bool = False
) -> Union[List[dict], Tuple[List[dict], dict]]:
    """
    Reads output of a TrajectoryTrainer run by frame. return_gp_data returns
    data about GP model growth useful for visualizing progress of model
    training.
    :param file: filename of output
    :param return_gp_data: flag for returning extra GP data
    :return: List of dictionaries with keys 'species', 'positions',
        'gp_forces', 'dft_forces', 'gp_stds', 'added_atoms', and
        'maes_by_species', optionally, gp_data dictionary
    """

    with open(file, "r") as f:
        lines = f.readlines()
        num_lines = len(lines)

    # Get indexes where frames begin, and include the index of the final line
    frame_indexes = [i for i in range(num_lines) if "-Frame:" in lines[i]] + [num_lines]

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

                frame_positions.append(
                    [float(split[1]), float(split[2]), float(split[3])]
                )
                gp_forces.append([float(split[4]), float(split[5]), float(split[6])])
                stds.append([float(split[7]), float(split[8]), float(split[9])])

                dft_forces.append(
                    [float(split[10]), float(split[11]), float(split[12])]
                )

            # Terminate at blank line between results
            else:
                break
        # Loop through information in frame after Data
        for i in range(
            frame_indexes[n] + len(frame_positions) + 2, frame_indexes[n + 1]
        ):

            if "Adding atom(s)" in lines[i]:
                # Splitting to target the 'added atoms' substring
                split_line = lines[i][15:-21]
                added_atoms = json.loads(split_line.strip())

            if "type " in lines[i]:
                cur_line = lines[i].split()
                frame_species_maes[cur_line[1]] = float(cur_line[3])

        cur_frame_stats = {
            "species": frame_atoms,
            "positions": frame_positions,
            "gp_forces": gp_forces,
            "dft_forces": dft_forces,
            "gp_stds": stds,
            "added_atoms": added_atoms,
            "maes_by_species": frame_species_maes,
        }

        frames.append(cur_frame_stats)

    if not return_gp_data:
        return frames

    # Compute information about GP training
    # to study GP growth and performance over trajectory

    gp_stats_line = [
        line for line in lines[:30] if "GP Statistics" in line and "Pre-run" not in line
    ][0][15:].strip()

    initial_gp_statistics = json.loads(gp_stats_line)

    # Get pre_run statistics (if pre-run was done):
    pre_run_gp_statistics = None
    pre_run_gp_stats_line = [line for line in lines if "Pre-run GP" in line]
    if pre_run_gp_stats_line:
        pre_run_gp_statistics = json.loads(pre_run_gp_stats_line[0][22:].strip())

    # Compute cumulative GP size
    cumulative_gp_size = [int(initial_gp_statistics["N"])]

    if pre_run_gp_stats_line:
        cumulative_gp_size.append(int(pre_run_gp_statistics["N"]))

    running_total = cumulative_gp_size[-1]

    for frame in frames:

        added_atom_dict = frame["added_atoms"]
        for val in added_atom_dict.values():
            running_total += len(val)
        cumulative_gp_size.append(running_total)

    # Compute MAEs for each element over time
    all_species = set()
    for frame in frames:
        all_species = all_species.union(set(frame["species"]))

    all_species = list(all_species)
    mae_by_elt = {elt: [] for elt in all_species}

    for frame in frames:
        for elt in all_species:
            cur_mae = frame["maes_by_species"].get(elt, np.nan)
            mae_by_elt[elt].append(cur_mae)

    gp_data = {
        "init_stats": initial_gp_statistics,
        "pre_train_stats": pre_run_gp_statistics,
        "cumulative_gp_size": cumulative_gp_size,
        "mae_by_elt": mae_by_elt,
    }

    return frames, gp_data
