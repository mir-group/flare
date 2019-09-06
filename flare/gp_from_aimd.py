"""
Tool to enable the development of a GP model based on an AIMD
trajectory. Contains methods to transfer the model to an OTF run /
MD engine run.
"""
import time

from flare.predict import predict_on_structure, \
    predict_on_structure_par, predict_on_structure_en, \
    predict_on_structure_par_en
from typing import List
from flare.struc import Structure
from flare.gp import GaussianProcess
import numpy as np
from copy import deepcopy
import pickle
import flare.output as output


class TrajectoryTrainer(object):

    def __init__(self, frames: List[Structure],
                 gp: GaussianProcess,
                 rel_std_tolerance: float = 1,
                 abs_std_tolerance: float = 1,
                 parallel: bool = False,
                 skip: int = 0,
                 calculate_energy: bool = False,
                 output_name: str = 'gp_from_aimd.out',
                 max_atoms_added: int = 0, max_trains: int = 10,
                 n_cpus: int = 1, shuffle_frames: bool = False,
                 verbose: int = 0, model_write: str = 'gp_model.pickle'):
        """
        Class which trains a GP off of an AIMD trajectory, and generates
        error statistics between the DFT and GP calls.

        :param gp:
        :param std_tolerance_factor:
        :param parallel:
        :param skip:
        :param calculate_energy:
        :param output_name:
        :param max_atoms_added:
        :param freeze_hyps:
        :param rescale_steps:
        :param rescale_temps:
        :param n_cpus:
        """

        self.frames = frames
        if shuffle_frames:
            np.random.shuffle(frames)
        self.gp = gp
        self.rel_std_tolerance = rel_std_tolerance
        self.abs_std_tolerance = abs_std_tolerance
        self.skip = skip
        self.dft_step = True
        self.max_trains = max_trains
        self.curr_step = 0
        self.max_atoms_added = max_atoms_added
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

        self.output_name = output_name

        # set number of cpus for parallelization
        self.n_cpus = n_cpus

        # To later be filled in using the time library
        self.start_time = None
        self.pickle_name = model_write

    def pre_run(self):
        """
        Various tasks to set up the AIMD training before commencing
        the run through the AIMD trajectory
        :return:
        """

        output.write_header(self.gp.cutoffs,
                            self.gp.kernel_name,
                            self.gp.hyps,
                            self.gp.algo,
                            dt=0,
                            Nsteps=len(self.frames),
                            structure=self.frames[0],
                            std_tolerance=(self.rel_std_tolerance,
                                           self.abs_std_tolerance),
                            output_name=self.output_name)

        self.start_time = time.time()

    def run(self):
        """
        Loop through frames and record the error between
        the GP predictions and the ground-truth forces. Train the GP and update
        the training set upon the triggering of the uncertainty threshold.
        :return:
        """

        self.pre_run()

        cur_frame = self.frames[0]
        train_atoms = []

        # First frame and no training set ("blank slate" run):
        # Take one of each atom species in the first frame
        # so all atomic species are represented.

        if len(self.gp.training_data) == 0:

            for unique_spec in set(cur_frame.coded_species):
                atoms_of_specie = cur_frame.indices_of_specie(unique_spec)
                train_atoms.append(atoms_of_specie[0])

            self.update_gp_and_print(cur_frame, train_atoms, train=True)

        if len(self.gp.training_data) > 0 and self.gp.l_mat is None:
            self.gp.train(monitor=self.verbose)

        # Loop through trajectory
        for i, cur_frame in enumerate(self.frames):

            if self.verbose >= 2: print("=====NOW ON FRAME {}=====".format(i))
            dft_forces = deepcopy(cur_frame.forces)
            self.pred_func(cur_frame, self.gp)

            # Convert to meV/A
            mae = np.mean(np.abs(cur_frame.forces - dft_forces))*1000
            mac = np.mean(np.abs(dft_forces))*1000

            output.write_gp_dft_comparison(
                curr_step=i, frame=cur_frame,
                start_time=time.time(), output_name=self.output_name,
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


        output.conclude_run(self.output_name)

    def update_gp_and_print(self, frame, train_atoms: List[int], train=True):
        """
        Update the internal GP model training set with a list of training
        atoms indexing atoms within the frame. If train is True, re-train
        the GP by optimizing hyperparameters.
        :param frame:
        :param train_atoms:
        :param train:
        :return:
        """

        output.write_to_output('\nAdding atom(s) {} to the '
                               'training set.\n'
                               .format(train_atoms, ),
                               self.output_name)
        output.write_to_output('Uncertainties: {}.\n'
                               .format(frame.stds[train_atoms]),
                               self.output_name)

        # update gp model
        self.gp.update_db(frame, frame.forces, custom_range=train_atoms)
        self.gp.set_L_alpha()
        # TODO double check that these are being called at the right time.
        if train:
            self.train_gp()

    def train_gp(self):
        self.gp.train(monitor=True if self.verbose >= 2 else False)

        output.write_hyps(self.gp.hyp_labels, self.gp.hyps,
                          self.start_time, self.output_name,
                          self.gp.like, self.gp.like_grad)
        self.train_count += 1

    def is_std_in_bound(self, frame):

        # This indicates test mode, as the GP is not being modified in any way
        if self.rel_std_tolerance == 0 and self.abs_std_tolerance ==0:
            return True,[-1]

        # set uncertainty threshold
        if self.rel_std_tolerance == 0:
            threshold = self.abs_std_tolerance
        elif self.abs_std_tolerance == 0 :
            threshold = self.rel_std_tolerance * np.abs(self.gp.hyps[-1])
        else:
            threshold = min(self.rel_std_tolerance * np.abs(self.gp.hyps[-1]),
                        self.abs_std_tolerance)

        # sort max stds
        max_stds = np.zeros((frame.nat))
        for atom_idx, std in enumerate(frame.stds):
            max_stds[atom_idx] = np.max(std)
        stds_sorted = np.argsort(max_stds)
        target_atoms = list(stds_sorted[-self.max_atoms_added:])

        # if above threshold, return atom
        if max_stds[stds_sorted[-1]] > threshold:
            return False, target_atoms
        else:
            return True, [-1]

    def write_model(self):
        pickle.dump(self.gp, self.model_write)
        # TODO serialize better
