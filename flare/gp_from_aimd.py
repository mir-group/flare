"""
Tool to enable the development of a GP model based on an AIMD
trajectory. Contains methods to transfer the model to an OTF run /
MD engine run.
"""
import time

from flare.predict import predict_on_structure, \
    predict_on_structure_par, predict_on_structure_en, \
    predict_on_structure_par_en
from typing import List, Tuple
from flare.struc import Structure, get_unique_species
from flare.gp import GaussianProcess
from flare.env import AtomicEnvironment
import numpy as np
from copy import deepcopy
import pickle
from flare.output import Output


class TrajectoryTrainer(object):

    def __init__(self, frames: List[Structure],
                 gp: GaussianProcess,
                 rel_std_tolerance: float = 1,
                 abs_std_tolerance: float = 1,
                 parallel: bool = False,
                 skip: int = 0,
                 calculate_energy: bool = False,
                 output_name: str = 'gp_from_aimd',
                 max_atoms_from_frame: int = np.inf, max_trains: int = np.inf,
                 min_atoms_added: int = 1,
                 n_cpus: int = 1, shuffle_frames: bool = False,
                 verbose: int = 0, model_write: str = '',
                 pre_train_on_skips: bool = False,
                 pre_train_seed_frames: List[Structure] = None,
                 pre_train_seed_envs: List[Tuple[AtomicEnvironment,
                                                 np.array]] = None,
                 pre_train_atoms_per_element: dict = None):
        """
        Class which trains a GP off of an AIMD trajectory, and generates
        error statistics between the DFT and GP calls.

        :param frames: List of structures to evaluate / train GP on
        :param gp: Gaussian Process object
        :param rel_std_tolerance: Train if uncertainty is above this *
        noise variance hyperparameter
        :param abs_std_tolerance: Train if uncertainty is above this
        :param parallel: Use parallel functions or not
        :param skip: Skip through frames
        :param calculate_energy: Use local energy kernel or not
        :param output_name: Write output of training to this file
        :param max_atoms_from_frame: Largest # of atoms added from one frame
        :param min_atoms_added: Only train when this many atoms have been added
        :param max_trains: Stop training GP after this many calls to train
        :param n_cpus: Number of CPUs to parallelize over
        :param shuffle_frames: Randomize order of frames for better training
        :param verbose: 0: Silent, 1: Minimal, 2: Lots of information
        :param model_write: Write output model here
        :param pre_train_on_skips: Train model on every n frames before running
        :param pre_train_seed_frames: Frames to train on before running
        :param pre_train_seed_envs: Environments to train on before running
        :param pre_train_atoms_per_element: Max # of environments to add from
        each species in the seed pre-training steps
        """

        self.frames = frames
        if shuffle_frames:
            np.random.shuffle(frames)
        self.gp = gp
        self.rel_std_tolerance = rel_std_tolerance
        self.abs_std_tolerance = abs_std_tolerance
        self.skip = skip
        self.max_trains = max_trains
        self.curr_step = 0
        self.max_atoms_from_frame = max_atoms_from_frame
        self.min_atoms_added = min_atoms_added
        self.verbose = verbose
        self.train_count = 0

        self.parallel = parallel

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

        self.output = Output(output_name)

        # set number of cpus for parallelization
        self.n_cpus = n_cpus

        # To later be filled in using the time library
        self.start_time = None
        self.pickle_name = model_write

        self.pre_train_on_skips = pre_train_on_skips
        self.seed_envs = [] if pre_train_seed_envs is None else \
            pre_train_seed_envs
        self.seed_frames = [] if pre_train_seed_frames is None \
            else pre_train_seed_frames
        self.pre_train_env_per_species = {} if pre_train_atoms_per_element \
                                               is None else pre_train_atoms_per_element

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

        # If seed environments were passed in, add them to the GP.
        for point in self.seed_envs:
            self.gp.add_one_env(point[0], point[1], train=False)

        # No training set ("blank slate" run) and no seeds specified:
        # Take one of each atom species in the first frame
        # so all atomic species are represented in the first step.
        # Otherwise use the seed frames passed in by user.
        if len(self.gp.training_data) == 0 and self.seed_frames is None:
            self.seed_frames = [self.frames[0]]

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

            self.update_gp_and_print(frame, train_atoms, train=False)

        # These conditions correspond to if either the GP was never trained
        # or if data was added to it during the pre-run.

        if (self.gp.l_mat is None) \
                or (self.seed_frames is not None
                    or self.seed_envs is not None):
            self.gp.train(output=self.output if self.verbose > 0 else None)

    def run(self):
        """
        Loop through frames and record the error between
        the GP predictions and the ground-truth forces. Train the GP and update
        the training set upon the triggering of the uncertainty threshold.
        :return:
        """

        self.pre_run()

        # Loop through trajectory
        for i, cur_frame in enumerate(self.frames):

            if self.verbose >= 2:
                print("=====NOW ON FRAME {}=====".format(i))
            dft_forces = deepcopy(cur_frame.forces)
            self.pred_func(cur_frame, self.gp)

            # Convert to meV/A
            mae = np.mean(np.abs(cur_frame.forces - dft_forces)) * 1000
            mac = np.mean(np.abs(dft_forces)) * 1000

            self.output.write_gp_dft_comparison(
                curr_step=i, frame=cur_frame,
                start_time=time.time(),
                dft_forces=dft_forces,
                mae=mae, mac=mac, local_energies=None)

            # get max uncertainty atoms
            std_in_bound, train_atoms = self.is_std_in_bound(cur_frame)
            if not std_in_bound:

                # compute mae and write to output
                # add max uncertainty atoms to training set
                self.update_gp_and_print(cur_frame, train_atoms, train=False)

                if self.train_count < self.max_trains:
                    self.train_gp()

        self.output.conclude_run()

        if self.pickle_name:
            with open(self.pickle_name, 'wb') as f:
                pickle.dump(self.gp, f)

    def update_gp_and_print(self, frame: Structure, train_atoms: List[int],
                            train: bool=True):
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
                                 .format(frame.stds[train_atoms]))

        # update gp model
        self.gp.update_db(frame, frame.forces, custom_range=train_atoms)
        self.gp.set_L_alpha()

        if train:
            self.train_gp()

    def train_gp(self):
        """
        Train the Gaussian process and write the results to the output file.
        """
        self.gp.train(output=self.output if self.verbose >= 2 else None)

        self.output.write_hyps(self.gp.hyp_labels, self.gp.hyps,
                               self.start_time,
                               self.gp.like, self.gp.like_grad)
        self.train_count += 1

    def is_std_in_bound(self, frame: Structure)->(bool, List[int]):
        """
        If the predicted variance is too high, returns a list of atoms
        with the highest uncertainty
        :param frame: Structure
        :return:
        """

        # This indicates test mode, as the GP is not being modified in any way
        if self.rel_std_tolerance == 0 and self.abs_std_tolerance == 0:
            return True, [-1]

        # set uncertainty threshold
        if self.rel_std_tolerance == 0:
            threshold = self.abs_std_tolerance
        elif self.abs_std_tolerance == 0:
            threshold = self.rel_std_tolerance * np.abs(self.gp.hyps[-1])
        else:
            threshold = min(self.rel_std_tolerance * np.abs(self.gp.hyps[-1]),
                            self.abs_std_tolerance)

        # sort max stds
        max_stds = np.zeros(frame.nat)
        for atom_idx, std in enumerate(frame.stds):
            max_stds[atom_idx] = np.max(std)
        stds_sorted = np.argsort(max_stds)

        # Handle case where unlimited atoms are added
        # or if max # of atoms exceeds size of frame
        if self.max_atoms_from_frame == np.inf or \
                self.max_atoms_from_frame > len(frame):
            target_atoms = list(stds_sorted)
        else:
            target_atoms = list(stds_sorted[-self.max_atoms_from_frame:])

        # if above threshold, return atom
        if max_stds[stds_sorted[-1]] > threshold:
            return False, target_atoms
        else:
            return True, [-1]
