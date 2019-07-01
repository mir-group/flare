import sys
import numpy as np
import datetime
import time
from typing import List
import copy
import multiprocessing as mp
import concurrent.futures
from flare import struc, gp, env, qe_util, md


class ActiveGp:
    def __init__(self, position_list, force_list, species, cell,
                 gp: gp.GaussianProcess,
                 std_tolerance_factor: float = 1, par: bool=False,
                 init_atoms: List[int]=None,
                 calculate_energy=False, output_name='otf_run.out',
                 max_atoms_added=1, freeze_hyps=10, no_cpus=1):

        self.position_list = position_list
        self.force_list = force_list
        self.species = species
        self.cell = cell
        self.gp = gp
        self.std_tolerance = std_tolerance_factor
        self.dft_step = True
        self.freeze_hyps = freeze_hyps

        positions = position_list[0]
        self.structure = struc.Structure(cell=cell, species=species,
                                         positions=positions)

        self.noa = self.structure.positions.shape[0]
        self.atom_list = list(range(self.noa))
        self.curr_step = 0

        self.dt = 1.0
        self.number_of_steps = 100
        self.temperature = 1.0
        self.KE = 1.0
        self.velocities = np.zeros((self.noa, 3))

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
        if not par and not calculate_energy:
            self.pred_func = self.predict_on_structure
        elif par and not calculate_energy:
            self.pred_func = self.predict_on_structure_par
        elif not par and calculate_energy:
            self.pred_func = self.predict_on_structure_en
        elif par and calculate_energy:
            self.pred_func = self.predict_on_structure_par_en
        self.par = par

        self.output_name = output_name

        # set number of cpus for qe runs
        self.no_cpus = no_cpus

    def run(self):
        write_header(self.gp.cutoffs, self.gp.kernel_name, self.gp.hyps,
                     self.gp.algo, self.dt, self.number_of_steps,
                     self.structure, self.output_name,
                     self.std_tolerance)

        self.start_time = time.time()

        for positions, forces in zip(self.position_list, self.force_list):

            # run DFT and train initial model if first step and DFT is on
            if self.curr_step == 0 and self.std_tolerance != 0:
                # update positions and forces
                self.structure.positions = positions
                self.structure.wrap_positions()
                self.structure.forces = copy.deepcopy(forces)
                self.dft_step = True
                self.record_state()

                # make initial gp model and predict forces
                self.update_gp(self.init_atoms, forces)
                self.dft_count += 1
                if (self.dft_count-1) < self.freeze_hyps:
                    self.train_gp()

            # after step 1, try predicting with GP model
            else:
                # update positions and forces
                self.structure.positions = positions
                self.structure.wrap_positions()
                self.structure.forces = copy.deepcopy(forces)
                self.dft_step = True
                self.record_state()

                # predict with gp
                self.pred_func()

                # get max uncertainty atoms
                std_in_bound, target_atoms = self.is_std_in_bound()

                # record GP forces
                self.dft_step = False
                self.record_state()

                mae = np.mean(np.abs(self.structure.forces - forces))
                mac = np.mean(np.abs(forces))
                write_to_output('\nmae: %.4f \n' % mae, self.output_name)
                write_to_output('\nmac: %.4f \n' % mac, self.output_name)

                # add max uncertainty atoms to training set
                if not std_in_bound:
                    self.update_gp(target_atoms, forces)
                    self.dft_count += 1
                    if (self.dft_count-1) < self.freeze_hyps:
                        self.train_gp()

            self.curr_step += 1

        conclude_run(self.output_name)

    def predict_on_atom(self, atom):
        chemenv = env.AtomicEnvironment(self.structure, atom, self.gp.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.abs(var)))

        return comps, stds

    def predict_on_atom_en(self, atom):
        chemenv = env.AtomicEnvironment(self.structure, atom, self.gp.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.abs(var)))

        # predict local energy
        local_energy = self.gp.predict_local_energy(chemenv)
        return comps, stds, local_energy

    def predict_on_structure_par(self):
        n = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.predict_on_atom, self.atom_list):
                for i in range(3):
                    self.structure.forces[n][i] = res[0][i]
                    self.structure.stds[n][i] = res[1][i]
                n += 1

    def predict_on_structure_par_en(self):
        n = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.predict_on_atom_en, self.atom_list):
                for i in range(3):
                    self.structure.forces[n][i] = res[0][i]
                    self.structure.stds[n][i] = res[1][i]
                self.local_energies[n] = res[2]
                n += 1

    def predict_on_structure(self):
        for n in range(self.structure.nat):
            chemenv = env.AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.abs(var))

    def predict_on_structure_en(self):
        for n in range(self.structure.nat):
            chemenv = env.AtomicEnvironment(self.structure, n, self.gp.cutoffs)
            for i in range(3):
                force, var = self.gp.predict(chemenv, i + 1)
                self.structure.forces[n][i] = float(force)
                self.structure.stds[n][i] = np.sqrt(np.abs(var))
            self.local_energies[n] = self.gp.predict_local_energy(chemenv)

    def update_gp(self, train_atoms, dft_frcs):
        write_to_output('\nAdding atom {} to the training set.\n'
                        .format(train_atoms),
                        self.output_name)
        write_to_output('Uncertainty: {}.\n'
                        .format(self.structure.stds[train_atoms[0]]),
                        self.output_name)

        # update gp model
        self.gp.update_db(self.structure, dft_frcs,
                          custom_range=train_atoms)

        self.gp.set_L_alpha()

    def train_gp(self):
        self.gp.train(True)
        write_hyps(self.gp.hyp_labels, self.gp.hyps,
                   self.start_time, self.output_name,
                   self.gp.like, self.gp.like_grad)

    def is_std_in_bound(self):
        # set uncertainty threshold
        if self.std_tolerance == 0:
            return True, -1
        elif self.std_tolerance > 0:
            threshold = self.std_tolerance * np.abs(self.gp.hyps[-1])
        else:
            threshold = np.abs(self.std_tolerance)

        # sort max stds
        max_stds = np.zeros((self.noa))
        for atom, std in enumerate(self.structure.stds):
            max_stds[atom] = np.max(std)
        stds_sorted = np.argsort(max_stds)
        target_atoms = list(stds_sorted[-self.max_atoms_added:])

        # if above threshold, return atom
        if max_stds[stds_sorted[-1]] > threshold:
            return False, target_atoms
        else:
            return True, [-1]

    def record_state(self):
        write_md_config(self.dt, self.curr_step, self.structure,
                        self.temperature, self.KE,
                        self.local_energies, self.start_time,
                        self.output_name, self.dft_step,
                        self.velocities)


def write_to_output(string: str, output_file: str = 'otf_run.out'):
    with open(output_file, 'a') as f:
        f.write(string)


def write_header(cutoffs, kernel_name, hyps, algo, dt, Nsteps, structure,
                 output_name, std_tolerance):

    with open(output_name, 'w') as f:
        f.write(str(datetime.datetime.now()) + '\n')

    if std_tolerance < 0:
        std_string = \
            'uncertainty tolerance: {} eV/A\n'.format(np.abs(std_tolerance))
    elif std_tolerance > 0:
        std_string = \
            'uncertainty tolerance: {} times noise \n'\
            .format(np.abs(std_tolerance))
    else:
        std_string = ''

    headerstring = ''
    headerstring += 'cutoffs: {}\n'.format(cutoffs)
    headerstring += 'kernel: {}\n'.format(kernel_name)
    headerstring += 'number of hyperparameters: {}\n'.format(len(hyps))
    headerstring += 'hyperparameters: {}' \
                    '\n'.format(hyps)
    headerstring += 'hyperparameter optimization algorithm: {}' \
                    '\n'.format(algo)
    headerstring += std_string
    headerstring += 'timestep (ps): {}\n'.format(dt)
    headerstring += 'number of atoms: {}\n'.format(structure.nat)
    headerstring += 'system species: {}\n'.format(set(structure.species))
    headerstring += 'periodic cell: \n'
    headerstring += str(structure.cell)

    headerstring += '-' * 80 + '\n'

    write_to_output(headerstring, output_name)


def write_md_config(dt, curr_step, structure, temperature, KE,
                    local_energies, start_time, output_name, dft_step,
                    velocities):
    string = ''

    # Mark if a frame had DFT forces with an asterisk
    if not dft_step:
        string += "\n-Frame: " + str(curr_step)
    else:
        string += '-' * 80 + '\n'
        string += "\n*-Frame: " + str(curr_step)

    string += '\nSimulation Time: %.3f ps \n' % (dt * curr_step)

    # Construct Header line
    string += 'El  Position (A) \t\t\t\t '
    if not dft_step:
        string += 'GP Force (ev/A) '
    else:
        string += 'DFT Force (ev/A) '
    string += '\t\t\t\t Std. Dev (ev/A) \n'

    # Construct atom-by-atom description
    for i in range(len(structure.positions)):
        string += structure.species[i] + ' '
        for j in range(3):
            string += str("%.8f" % structure.positions[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8f" % structure.forces[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % structure.stds[i][j]) + ' '
        string += '\n'

    string += '\n'

    # calculate potential and total energy
    if local_energies is not None:
        pot_en = np.sum(local_energies)
        tot_en = KE + pot_en
        string += \
            'potential energy: %.6f eV \n' % pot_en
        string += 'total energy: %.6f eV \n' % tot_en

    string += 'wall time from start: %.2f s \n' % \
        (time.time() - start_time)

    write_to_output(string, output_name)


def write_hyps(hyp_labels, hyps, start_time, output_name, like, like_grad):
    write_to_output('\nGP hyperparameters: \n', output_name)

    for i, label in enumerate(hyp_labels):
        write_to_output('Hyp{} : {} = {}\n'.format(i, label, hyps[i]),
                        output_name)

    write_to_output('likelihood: '+str(like)+'\n', output_name)
    write_to_output('likelihood gradient: '+str(like_grad)+'\n', output_name)
    time_curr = time.time() - start_time
    write_to_output('wall time from start: %.2f s \n' % time_curr,
                    output_name)


def conclude_run(output_name):
    footer = '-' * 20 + '\n'
    footer += 'Run complete. \n'

    write_to_output(footer, output_name)
