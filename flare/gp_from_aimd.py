"""
Tool to enable the development of a GP model based on an AIMD
trajectory. Contains methods to transfer the model to an OTF run /
MD engine run.
"""
import time
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.output import Output
from flare.predict import predict_on_structure, \
    predict_on_structure_par, predict_on_structure_en, \
    predict_on_structure_par_en
from flare.struc import Structure
from flare.util import element_to_Z, \
    is_std_in_bound_per_species, is_force_in_bound_per_species


class TrajectoryTrainer(object):

    def __init__(self, frames: List[Structure],
                 gp: GaussianProcess,
                 rel_std_tolerance: float = 1,
                 abs_std_tolerance: float = 1,
                 abs_force_tolerance: float = 0,
                 max_force_error: float = np.inf,
                 parallel: bool = False,
                 no_cpus: int = None,
                 skip: int = 1,
                 validate_ratio: float = 0.1,
                 calculate_energy: bool = False,
                 output_name: str = 'gp_from_aimd',
                 pre_train_max_iter: int = 50,
                 max_atoms_from_frame: int = np.inf, max_trains: int = np.inf,
                 min_atoms_added: int = 1, shuffle_frames: bool = False,
                 verbose: int = 0,
                 pre_train_on_skips: int = -1,
                 pre_train_seed_frames: List[Structure] = None,
                 pre_train_seed_envs: List[Tuple[AtomicEnvironment,
                                                 np.array]] = None,
                 pre_train_atoms_per_element: dict = None,
                 train_atoms_per_element: dict = None,
                 checkpoint_interval: int = None,
                 model_format: str = 'json'):
        """
        Class which trains a GP off of an AIMD trajectory, and generates
        error statistics between the DFT and GP calls.

        :param frames: List of structures to evaluate / train GP on
        :param gp: Gaussian Process object
        :param rel_std_tolerance: Train if uncertainty is above this *
        noise variance hyperparameter
        :param abs_std_tolerance: Train if uncertainty is above this
        :param abs_force_tolerance: Add atom force error exceeds this
        :param max_force_error: Don't add atom if force error exceeds this
        :param parallel: Use parallel functions or not
        :param validate_ratio: Fraction of frames used for validation
        :param no_cpus: number of cpus to run with multithreading
        :param skip: Skip through frames
        :param calculate_energy: Use local energy kernel or not
        :param output_name: Write output of training to this file
        :param max_atoms_from_frame: Largest # of atoms added from one frame
        :param min_atoms_added: Only train when this many atoms have been added
        :param max_trains: Stop training GP after this many calls to train
        :param n_cpus: Number of CPUs to parallelize over
        :param shuffle_frames: Randomize order of frames for better training
        :param verbose: 0: Silent, 1: Minimal, 2: Lots of information
        :param pre_train_on_skips: Train model on every n frames before running
        :param pre_train_seed_frames: Frames to train on before running
        :param pre_train_seed_envs: Environments to train on before running
        :param pre_train_atoms_per_element: Max # of environments to add from
        each species in the seed pre-training steps
        :param train_atoms_per_element: Max # of environments to add from
        each species in the training steps
        :param checkpoint_interval: How often to write model after trainings
        :param model_format: Format to write GP model to
        """

        self.frames = frames
        if shuffle_frames:
            np.random.shuffle(frames)
        self.gp = gp
        self.rel_std_tolerance = rel_std_tolerance
        self.abs_std_tolerance = abs_std_tolerance
        self.abs_force_tolerance = abs_force_tolerance
        self.max_force_error = max_force_error

        self.skip = skip
        assert (skip >= 1), "skip needs to be an integer >= 1"
        self.validate_ratio = validate_ratio
        self.max_trains = max_trains
        self.curr_step = 0
        self.max_atoms_from_frame = max_atoms_from_frame
        self.min_atoms_added = min_atoms_added
        self.verbose = verbose
        self.train_count = 0

        self.parallel = parallel
        self.no_cpus = no_cpus

        # set pred function
        if parallel:
            if calculate_energy:
                self.pred_func = predict_on_structure_par_en
            else:
                self.pred_func = predict_on_structure_par
        else:
            if calculate_energy:
                self.pred_func = predict_on_structure_en
            else:
                self.pred_func = predict_on_structure

        self.output = Output(output_name, always_flush=True)

        # To later be filled in using the time library
        self.start_time = None

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
        self.checkpoint_interval = checkpoint_interval
        self.model_format = model_format
        self.output_name = output_name

    def pre_run(self):
        """
        Various tasks to set up the AIMD training before commencing
        the run through the AIMD trajectory.
        1. Print the output.
        2. Pre-train the GP with the seed frames and
        environments. If no seed frames or environments and the GP has no
        training set, then seed with at least one atom from each
        """

        self.output.write_header(self.gp.cutoffs,
                                 self.gp.kernel_name,
                                 self.gp.hyps,
                                 self.gp.algo,
                                 dt=0,
                                 Nsteps=len(self.frames),
                                 structure=self.frames[0],
                                 std_tolerance=(self.rel_std_tolerance,
                                                self.abs_std_tolerance))

        self.start_time = time.time()
        if self.verbose >= 3:
            print("Now beginning pre-run activity.")
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
                    species_i, np.inf), self.max_atoms_from_frame)

                for atom in atoms_of_specie[:n_to_add]:
                    train_atoms.append(atom)
                    atom_count += 1

            self.update_gp_and_print(frame, train_atoms, train=False)

        if self.verbose >= 3 and atom_count > 0:
            print(f"Added {atom_count} atoms to pretrain")

        if self.seed_envs or atom_count or self.seed_frames:
            if self.verbose >= 3:
                print("Now commencing pre-run training of GP (which has "
                      "non-empty training set)")
            self.train_gp(max_iter=self.pre_train_max_iter)

    def run(self):
        """
        Loop through frames and record the error between
        the GP predictions and the ground-truth forces. Train the GP and update
        the training set upon the triggering of the uncertainty threshold.
        :return:
        """
        if self.verbose >= 3:
            print("Commencing run with pre-run...")
        self.pre_run()

        if self.validate_ratio > 0:
            train_frame = int(len(self.frames) * (1 - self.validate_ratio))
        else:
            train_frame = len(self.frames)

        # Loop through trajectory
        nsample = 0
        for i, cur_frame in enumerate(self.frames[::self.skip]):

            if self.verbose >= 2:
                print("=====NOW ON FRAME {}=====".format(i))
            dft_forces = deepcopy(cur_frame.forces)

            self.pred_func(cur_frame, self.gp, self.no_cpus)

            # Convert to meV/A
            error = np.abs(cur_frame.forces - dft_forces)

            self.output.write_gp_dft_comparison(
                curr_step=i, frame=cur_frame,
                start_time=time.time(),
                dft_forces=dft_forces,
                error=error,
                local_energies=None)

            if i < train_frame:
                # Get max uncertainty atoms
                std_in_bound, std_train_atoms = is_std_in_bound_per_species(
                    rel_std_tolerance=self.rel_std_tolerance,
                    abs_std_tolerance=self.abs_std_tolerance,
                    noise=self.gp.hyps[-1], structure=cur_frame,
                    max_atoms_added=self.max_atoms_from_frame,
                    max_by_species=self.train_env_per_species)

                # Get max force error atoms
                force_in_bound, force_train_atoms = \
                    is_force_in_bound_per_species(
                        abs_force_tolerance=self.abs_force_tolerance,
                        predicted_forces=cur_frame.forces,
                        label_forces=dft_forces,
                        structure=cur_frame,
                        max_atoms_added=self.max_atoms_from_frame,
                        max_by_species=self.train_env_per_species,
                        max_force_error=self.max_force_error)

                if (not std_in_bound) or (not force_in_bound):

                    train_atoms = list(set(std_train_atoms).union(
                        force_train_atoms) - {-1})

                    # Compute mae and write to output;
                    # Add max uncertainty atoms to training set
                    self.update_gp_and_print(
                        cur_frame, train_atoms, train=False)
                    nsample += len(train_atoms)
                    # Re-train if number of sampled atoms is high enough
                    if nsample >= self.min_atoms_added or (
                            i + 1) == train_frame:
                        if self.train_count < self.max_trains:
                            self.train_gp()
                        else:
                            self.gp.update_L_alpha()
                        nsample = 0
                    else:
                        self.gp.update_L_alpha()

                if (i + 1) == train_frame:
                    self.gp.check_L_alpha()

        self.output.conclude_run()

        if self.model_format:
            self.gp.write_model(self.output_name+'_model', self.model_format)

    def update_gp_and_print(self, frame: Structure, train_atoms: List[int],
                            train: bool = True):
        """
        Update the internal GP model training set with a list of training
        atoms indexing atoms within the frame. If train is True, re-train
        the GP by optimizing hyperparameters.
        :param frame: Structure to train on
        :param train_atoms: Index atoms to train on
        :param train: Train or not
        :return:
        """

        self.output.write_to_log('\nAdding atom(s) {} to the '
                                 'training set.\n'
                                 .format(train_atoms, ))
        self.output.write_to_log('Uncertainties: {}.\n'
                                 .format(frame.stds[train_atoms]),
                                 flush=True)

        # update gp model
        self.gp.update_db(frame, frame.forces, custom_range=train_atoms)

        if train:
            self.train_gp()

    def train_gp(self, max_iter: int = None):
        """
        Train the Gaussian process and write the results to the output file.
        """
        if self.verbose >= 1:
            self.output.write_to_log('Train GP\n')

        # TODO: Improve flexibility in GP training to make this next step
        # unnecessary, so maxiter can be passed as an argument

        # Don't train if maxiter == 0
        if max_iter == 0:
            self.gp.check_L_alpha()
        elif max_iter is not None:
            temp_maxiter = self.gp.maxiter
            self.gp.maxiter = max_iter
            self.gp.train(output=self.output if self.verbose >= 2 else None)
            self.gp.maxiter = temp_maxiter
        else:
            self.gp.train(output=self.output if self.verbose >= 2 else None)

        self.output.write_hyps(self.gp.hyp_labels, self.gp.hyps,
                               self.start_time,
                               self.gp.likelihood, self.gp.likelihood_gradient)
        self.train_count += 1

        if self.checkpoint_interval \
                and self.train_count % self.checkpoint_interval == 0 \
                and self.model_format:
            self.gp.write_model(self.output_name+'_model', self.model_format)
