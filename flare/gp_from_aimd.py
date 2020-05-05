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
particularly important and so you want a good representation in the training set,
then you would want to include as many as possible in the training set
during the seed part of the training.

Inversely, if a system has high representation of a species well-described
by a simple 2+3 body kernel, you may want it to be less well represented
in the seeded training set.

By specifying the pre_train_atoms_per_element, you can limit the number of
atoms of a given species which are added in. You can also limit the number
of atoms which are added from a given seed frame.

"""
import time
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
from math import inf
import warnings

from flare.env import AtomicEnvironment
from flare.gp import GaussianProcess
from flare.output import Output
from flare.predict import predict_on_atom, predict_on_atom_en, \
    predict_on_structure_par, predict_on_structure_par_en
from flare.struc import Structure
from flare.util import element_to_Z, \
    is_std_in_bound_per_species, is_force_in_bound_per_species, \
    Z_to_element, subset_of_frame_by_element
from flare.mgp.otf import predict_on_structure_mgp
from flare.mgp.mgp_en import MappedGaussianProcess


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
                 pre_train_max_iter: int = 50,
                 max_atoms_from_frame: int = np.inf,
                 max_trains: int = np.inf,
                 min_atoms_per_train: int = 1,
                 shuffle_frames: bool = False,
                 verbose: int = 1,
                 pre_train_on_skips: int = -1,
                 pre_train_seed_frames: List[Structure] = None,
                 pre_train_seed_envs: List[Tuple[AtomicEnvironment,
                                                 'np.array']] = None,
                 pre_train_atoms_per_element: dict = None,
                 train_atoms_per_element: dict = None,
                 predict_atoms_per_element: dict = None,
                 checkpoint_interval: int = 1,
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
        :param max_atoms_from_frame: Largest # of atoms added from one frame
        :param min_atoms_per_train: Only train when this many atoms have been
            added
        :param max_trains: Stop training GP after this many calls to train
        :param n_cpus: Number of CPUs to parallelize over for parallelization over atoms
        :param shuffle_frames: Randomize order of frames for better training
        :param verbose: 0: Silent, NO output written or printed at all.
                        1: Minimal,
                        2: Lots of information
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
        :param checkpoint_interval: How often to write model after trainings
        :param model_format: Format to write GP model to
        """

        # Set up parameters
        self.frames = frames
        if shuffle_frames:
            np.random.shuffle(frames)

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
        self.verbose = verbose
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
        assert (validate_ratio >= 0 and validate_ratio <= 1), \
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
        self.verbose = verbose
        if self.verbose:
            self.output = Output(output_name, always_flush=True)
        else:
            self.output = None
        self.checkpoint_interval = checkpoint_interval
        self.model_format = model_format
        self.output_name = output_name

        # Defining variables to be used later
        self.curr_step = 0
        self.train_count = 0
        self.start_time = None

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
            raise NotImplementedError("Pre-running not" \
                                      "yet configured for MGP")
        if self.verbose:
            self.output.write_header(self.gp.cutoffs,
                                     self.gp.kernel_name,
                                     self.gp.hyps,
                                     self.gp.opt_algorithm,
                                     dt=0,
                                     Nsteps=len(self.frames),
                                     structure=self.frames[0],
                                     std_tolerance=(self.rel_std_tolerance,
                                                    self.abs_std_tolerance),
                                     optional={
                                         'GP Statistics': self.gp.training_statistics})

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

        if self.verbose >= 3 and atom_count > 0:
            self.output.write_to_log(f"Added {atom_count} atoms to pretrain\n" \
                                     f"In total {len(self.gp.training_data)} atoms",
                                     flush=True)

        if (self.seed_envs or atom_count or self.seed_frames) and \
                (self.pre_train_max_iter or self.max_trains):
            if self.verbose >= 3:
                print("Now commencing pre-run training of GP (which has "
                      "non-empty training set)")
            self.train_gp(max_iter=self.pre_train_max_iter)
        else:
            if self.verbose >= 3:
                print("Now commencing pre-run set up of GP (which has "
                      "non-empty training set)")
            self.gp.set_L_alpha()

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
        if self.verbose >= 3:
            print("Commencing run with pre-run...")
        if not self.mgp:
            self.pre_run()

        # Past this frame, stop adding atoms to the training set
        #  (used for validation of model)
        train_frame = int(len(self.frames[::self.skip]) * (1 -
                                                           self.validate_ratio))

        # Loop through trajectory.
        nsample = 0
        for i, cur_frame in enumerate(self.frames[::self.skip]):

            if self.verbose >= 2:
                print(f"=====NOW ON FRAME {i}=====")

            # If no predict_atoms_per_element was specified, predict_atoms
            # will be equal to every atom in the frame.
            predict_atoms = subset_of_frame_by_element(cur_frame,
                                                       self.predict_atoms_per_element)
            # Atoms which are skipped will have NaN as their force / std values
            local_energies = None

            # Three different predictions: Either MGP, GP with energy,
            # or GP without
            if self.mgp:
                pred_forces, pred_stds = self.pred_func(
                    structure=cur_frame, mgp=self.gp, write_to_structure=False,
                    selective_atoms=predict_atoms, skipped_atom_value=
                    np.nan)
            elif self.calculate_energy:
                pred_forces, pred_stds, local_energies = self.pred_func(
                    structure=cur_frame, gp=self.gp, n_cpus=self.n_cpus,
                    write_to_structure=False, selective_atoms=predict_atoms,
                    skipped_atom_value=np.nan)
            else:
                pred_forces, pred_stds = self.pred_func(structure=cur_frame,
                                                gp=self.gp,
                                                n_cpus=self.n_cpus,
                                                write_to_structure=False,
                                                selective_atoms=predict_atoms,
                                                skipped_atom_value=np.nan)

            # Get Error
            dft_forces = cur_frame.forces
            error = np.abs(pred_forces - dft_forces)

            # Create dummy frame with the predicted forces written
            dummy_frame = deepcopy(cur_frame)
            dummy_frame.forces = pred_forces
            dummy_frame.stds = pred_stds

            if self.verbose:
                self.output.write_gp_dft_comparison(
                    curr_step=i, frame=dummy_frame,
                    start_time=time.time(),
                    dft_forces=dft_forces,
                    error=error,
                    local_energies=local_energies)

            if i < train_frame:
                # Noise hyperparameter & relative std tolerance is not for mgp.
                if self.mgp:
                    noise = 0
                else:
                    noise = self.gp.hyps[-1]
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

                    # Compute mae and write to output;
                    # Add max uncertainty atoms to training set
                    self.update_gp_and_print(
                        cur_frame, train_atoms=train_atoms,
                        uncertainties=pred_stds,
                        train=False)
                    nsample += len(train_atoms)
                    # Re-train if number of sampled atoms is high enough
                    if nsample >= self.min_atoms_per_train or (
                            i + 1) == train_frame:
                        if self.train_count < self.max_trains:
                            self.train_gp()
                        else:
                            self.gp.update_L_alpha()
                        nsample = 0
                    else:
                        self.gp.update_L_alpha()

                    if self.checkpoint_interval \
                            and self.train_count % self.checkpoint_interval == 0 \
                            and self.model_format:
                        self.gp.write_model(f'{self.output_name}_checkpt',
                                            self.model_format)

                if (i + 1) == train_frame and not self.mgp:
                    self.gp.check_L_alpha()

        if self.verbose:
            self.output.conclude_run()

        if self.model_format and not self.mgp:
            self.gp.write_model(f'{self.output_name}_model',
                                self.model_format)

    def update_gp_and_print(self, frame: Structure, train_atoms: List[int],
                            uncertainties: List[int]=None,
                            train: bool = True):
        """
        Update the internal GP model training set with a list of training
        atoms indexing atoms within the frame. If train is True, re-train
        the GP by optimizing hyperparameters.
        :param frame: Structure to train on
        :param train_atoms: Index atoms to train on
        :param: uncertainties: Uncertainties to print, pass in [] to silence
        :param train: Train or not
        :return: None
        """



        # Group added atoms by species for easier output
        added_species = [Z_to_element(frame.coded_species[at]) for at in
                         train_atoms]
        added_atoms = {spec: [] for spec in set(added_species)}

        for atom, spec in zip(train_atoms, added_species):
            added_atoms[spec].append(atom)


        if self.verbose:
            self.output.write_to_log(f'\nAdding atom(s) {added_atoms}'
                                     ' to the training set.\n')

        if uncertainties is None and not (uncertainties is []):
            uncertainties = frame.stds[train_atoms]

        if self.verbose and not(uncertainties is []):
            self.output.write_to_log(f'Uncertainties: '
                                     f'{uncertainties}.\n',
                                     flush=True)

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

        if self.verbose:
            self.output.write_hyps(self.gp.hyp_labels, self.gp.hyps,
                                   self.start_time,
                                   self.gp.likelihood,
                                   self.gp.likelihood_gradient,
                                   hyps_mask=self.gp.hyps_mask)
        self.train_count += 1
