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
from flare.learning_protocol import LearningProtocol


class TrajectoryTrainer(LearningProtocol):

    def __init__(self, gp: Union[GaussianProcess, MappedGaussianProcess],
                 active_frames: List[Structure] = None,
                 active_rel_var_tol: float = 4,
                 active_abs_var_tol: float = 1,
                 active_abs_error_tol: float = 0,
                 active_error_tol_cutoff: float = inf,
                 active_max_trains: int = np.inf,

                 predict_atoms_per_element: dict = None,


                 active_skip: int = 1,
                 shuffle_active_frames: bool = False,

                 validate_ratio: float = 0.0,
                 **kwargs):
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

        LearningProtocol.__init__(self, gp, **kwargs)

        self.predict_atoms_per_element = predict_atoms_per_element

        # Set up parameters
        self.frames = active_frames
        if shuffle_active_frames:
            np.random.shuffle(active_frames)

        # Parameters for negotiating with the training active_frames
        self.skip = active_skip
        assert (isinstance(active_skip, int) and active_skip >= 1), \
            "Skip needs to be a  positive integer."
        self.validate_ratio = validate_ratio
        assert (0 <= validate_ratio <= 1), \
            "validate_ratio needs to be [0,1]"

    def preparation_for_active_run(self):
        # Past this frame, stop adding atoms to the training set
        #  (used for validation of model)
        self.train_frame = int(len(self.frames[::self.skip])
                               * (1 - self.validate_ratio))
        self.frames = self.frames[::self.skip]

    def get_next_active_frame(self):
        self.curr_active_frame_index += 1
        if self.curr_active_frame_index < len(self.frames):
            return self.frames[self.curr_active_frame_index]
        return None

    def decide_to_update_db(self):
        if self.curr_active_frame_index <= self.train_frame \
                and (not self.mgp) \
                and len(self.gp) <= self.max_model_size:
            return True
        return False

    def decide_to_checkLalpha(self):
        if (self.curr_active_frame_index + 1) == self.train_frame and not self.mgp:
            return True
        return False

    def decide_to_train(self):

        if self.cur_atoms_added_train >= self.min_atoms_per_train or (
             self.curr_active_frame_index + 1) == self.train_frame:
            if self.train_count < self.max_trains:
                return True

        return False



def parse_trajectory_trainer_output(file: str, return_gp_data: bool = False) \
        -> Union[List[dict], Tuple[List[dict], dict]]:
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
                           'positions': frame_positions,
                           'gp_forces': gp_forces,
                           'dft_forces': dft_forces,
                           'gp_stds': stds,
                           'added_atoms': added_atoms,
                           'maes_by_species': frame_species_maes}

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
