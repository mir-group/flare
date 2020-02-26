"""
Class which contains various methods to print the output of different
ways of using FLARE, such as training a GP from an AIMD run,
or running an MD simulation updated on-the-fly.
"""
import datetime
import os
import shutil
import time

import multiprocessing
import numpy as np

from flare.util import Z_to_element


class Output:
    """
    This is an I/O class that hosts the log files for OTF and Trajectories
    class. It is also used in get_neg_like_grad and get_neg_likelihood in
    gp_algebra to print intermediate results.

    It opens and prints files with the basename prefix and different
    suffixes corresponding to different kinds of output data.

    :param basename: Base output file name, suffixes will be added
    :type basename: str, optional
    :param always_flush: Always write to file instantly
    :type always_flus: bool, optional
    """

    def __init__(self, basename: str = 'otf_run',
                 always_flush: bool = False):
        """
        Construction. Open files.
        """
        self.basename = f"{basename}"
        self.outfiles = {}
        filesuffix = {'log': '.out', 'hyps': '-hyps.dat'}

        for filetype in filesuffix.keys():
            self.open_new_log(filetype, filesuffix[filetype])

        self.always_flush = always_flush

    def conclude_run(self):
        """
        destruction function that closes all files
        """

        print('-' * 20, file=self.outfiles['log'])
        print('Run complete.', file=self.outfiles['log'])
        for (k, v) in self.outfiles.items():
            v.close()
        del self.outfiles
        self.outfiles = {}

    def open_new_log(self, filetype: str, suffix: str):
        """
        Open files.  If files with the same
        name are exist, they are backed up with a suffix "-bak".

        :param filetype: the key name in self.outfiles
        :param suffix: the suffix of the file to be opened
        """

        filename = self.basename + suffix

        # if the file exists, back up
        if os.path.isfile(filename):
            shutil.copy(filename, filename + "-bak")

        if filetype in self.outfiles.keys():
            if self.outfiles[filetype].closed:
                self.outfiles[filetype] = open(filename, "w+")
        else:
            self.outfiles[filetype] = open(filename, "w+")

    def write_to_log(self, logstring: str, name: str = "log",
                     flush: bool = False):
        """
        Write any string to logfile

        :param logstring: the string to write
        :param name: the key name of the file to print
        :param flush: whether it should be flushed
        """
        self.outfiles[name].write(logstring)

        if flush or self.always_flush:
            self.outfiles[name].flush()

    def write_header(self, cutoffs, kernel_name: str,
                     hyps, algo: str, dt: float,
                     Nsteps: int, structure,
                     std_tolerance,
                     optional: dict = None):
        """
        Write header to the log function

        :param cutoffs: GP cutoffs
        :param kernel_name: Kernel names
        :param hyps: list of hyper-parameters
        :param algo: algorithm for hyper parameter optimization
        :param dt: timestep for OTF MD
        :param Nsteps: total number of steps for OTF MD
        :param structure: the atomic structure
        :param std_tolerance: tolarence for active learning
        :param optional: a dictionary of all the other parameters
        """

        f = self.outfiles['log']
        f.write(f'{datetime.datetime.now()} \n')

        if isinstance(std_tolerance, tuple):
            std_string = 'relative uncertainty tolerance: ' \
                         f'{std_tolerance[0]} eV/A\n'
            std_string += 'absolute uncertainty tolerance: ' \
                          f'{std_tolerance[1]} eV/A\n'
        elif std_tolerance < 0:
            std_string = \
                f'uncertainty tolerance: {np.abs(std_tolerance)} eV/A\n'
        elif std_tolerance > 0:
            std_string = \
                f'uncertainty tolerance: {np.abs(std_tolerance)} times noise \n'
        else:
            std_string = ''

        headerstring = ''
        headerstring += \
            f'number of cpu cores: {multiprocessing.cpu_count()}\n'
        headerstring += f'cutoffs: {cutoffs}\n'
        headerstring += f'kernel: {kernel_name}\n'
        headerstring += f'number of hyperparameters: {len(hyps)}\n'
        headerstring += f'hyperparameters: {hyps}\n'
        headerstring += f'hyperparameter optimization algorithm: {algo}\n'
        headerstring += std_string
        headerstring += f'timestep (ps): {dt}\n'
        headerstring += f'number of frames: {Nsteps}\n'
        headerstring += f'number of atoms: {structure.nat}\n'
        headerstring += f'system species: {set(structure.species_labels)}\n'
        headerstring += 'periodic cell: \n'
        headerstring += str(structure.cell)

        if optional:
            for key, value in optional.items():
                headerstring += f"{key}: {value} \n"

        # report previous positions
        headerstring += '\nprevious positions (A):\n'
        for i in range(len(structure.positions)):
            headerstring += f'{structure.species_labels[i]:5}'
            for j in range(3):
                headerstring += f'{structure.prev_positions[i][j]:10.4f}'
            headerstring += '\n'
        headerstring += '-' * 80 + '\n'

        f.write(headerstring)

        if self.always_flush:
            f.flush()

    def write_md_config(self, dt, curr_step, structure,
                        temperature, KE, local_energies,
                        start_time, dft_step, velocities):
        """ write md configuration in log file

        :param dt: timestemp of OTF MD
        :param curr_step: current timestep of OTF MD
        :param structure: atomic structure
        :param temperature: current temperature
        :param KE: current total kinetic energy
        :param local_energies: local energy
        :param start_time: starting time for time profiling
        :param dft_step: # of DFT calls
        :param velocities: list of velocities

        :return:
        """

        string = ''

        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += '-' * 80 + '\n'
            string += f"-Frame: {curr_step} "
            header = "-"
        else:
            string += f"\n*-Frame: {curr_step} "
            header = "*-"

        string += f'\nSimulation Time: {(dt * curr_step):.3} ps \n'

        # Construct Header line
        n_space = 30
        string += str.ljust('El', 5)
        string += str.center('Position (A)', n_space)
        string += ' ' * 4
        if not dft_step:
            string += str.center('GP Force (ev/A)', n_space)
            string += ' ' * 4
        else:
            string += str.center('DFT Force (ev/A)', n_space)
            string += ' ' * 4
        string += str.center('Std. Dev (ev/A)', n_space) + ' ' * 4
        string += str.center('Velocities (A/ps)', n_space) + '\n'

        # Construct atom-by-atom description
        for i in range(len(structure.positions)):
            string += f'{structure.species_labels[i]:5}'
            # string += '\t'
            for j in range(3):
                string += f'{structure.positions[i][j]:10.4f}'
            string += ' ' * 4
            for j in range(3):
                string += f'{structure.forces[i][j]:10.4f}'
            string += ' ' * 4
            for j in range(3):
                string += f'{structure.stds[i][j]:10.4f}'
            string += ' ' * 4
            for j in range(3):
                string += f'{velocities[i][j]:10.4f}'
            string += '\n'

        string += '\n'
        string += f'temperature: {temperature:.2f} K \n'
        string += f'kinetic energy: {KE:.6f} eV \n'

        # calculate potential and total energy
        if local_energies is not None:
            pot_en = np.sum(local_energies)
            tot_en = KE + pot_en
            string += \
                f'potential energy: {pot_en:.6f} eV \n'
            string += f'total energy: {tot_en:.6f} eV \n'

        string += 'wall time from start: '
        string += f'{(time.time() - start_time):.2f} s \n'

        self.outfiles['log'].write(string)

        if self.always_flush:
            self.outfiles['log'].flush()

    def write_xyz(self, curr_step: int, pos: np.array, species: list,
                  filename: str,
                  header="",
                  forces: np.array = None, stds: np.array = None,
                  forces_2: np.array = None):
        """ write atomic configuration in xyz file

        :param curr_step: Int, number of frames to note in the comment line
        :param pos:       nx3 matrix of forces, positions, or nything
        :param species:   n element list of symbols
        :param filename:  file to print
        :param header:    header printed in comments
        :param forces: list of forces on atoms predicted by GP
        :param stds: uncertainties predicted by GP
        :param forces_2: true forces from ab initio source
        """

        natom = len(species)
        string = f'{natom}\n'

        # comment line
        # Mark if a frame had DFT forces with an asterisk
        string += f"{header} Frame: {curr_step}\n"

        # Construct atom-by-atom description
        for i in range(natom):
            string += f'{species[i]} '
            string += f'{pos[i, 0]:10.3} {pos[i, 1]:10.3} {pos[i, 2]:10.3}'

            if forces is not None and stds is not None and forces_2 is not \
                    None:

                string += f' {forces[i, 0]:10.3} {forces[i, 1]:10.3} ' \
                          f'{forces[i, 2]:10.3}'
                string += f' {stds[i, 0]:10.3} {stds[i, 1]:10.3} ' \
                          f'{stds[i, 2]:10.3}'
                string += f' {forces_2[i, 0]:10.3} {forces_2[i,1]:10.3} ' \
                          f'{forces_2[i, 2]:10.3}\n'

            else:
                string += '\n'
        self.outfiles[filename].write(string)

        if self.always_flush:
            self.outfiles[filename].flush()

    def write_xyz_config(self, curr_step, structure, dft_step,
                         forces: np.array = None, stds: np.array = None,
                         forces_2: np.array = None):
        """ write atomic configuration in xyz file

        :param curr_step: Int, number of frames to note in the comment line
        :param structure: Structure, contain positions and forces
        :param dft_step:  Boolean, whether this is a DFT call.
        :param forces: Optional list of forces to print in xyz file
        :param stds: Optional list of uncertanties to print in xyz file
        :param forces_2: Optional second list of forces (e.g. DFT forces)

        :return:
        """

        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            header = ""
        else:
            header = "*"
        self.write_xyz(curr_step=curr_step, pos=structure.positions,
                       species=structure.species_labels, filename='xyz',
                       header=header,
                       forces=forces, stds=stds, forces_2=forces_2)

    def write_hyps(self, hyp_labels, hyps, start_time, like, like_grad,
                   name='log'):
        """ write hyperparameters to logfile

        :param name:
        :param hyp_labels: labels for hyper-parameters. can be None
        :param hyps: list of hyper-parameters
        :param start_time: start time for time profiling
        :param like: likelihood
        :param like_grad: gradient of likelihood

        :return:
        """
        f = self.outfiles[name]
        f.write('\nGP hyperparameters: \n')

        if hyp_labels is not None:
            for i, label in enumerate(hyp_labels):
                f.write(f'Hyp{i} : {label} = {hyps[i]:.4f}\n')
        else:
            for i, hyp in enumerate(hyps):
                f.write(f'Hyp{i} : {hyp:.4f}\n')

        f.write(f'likelihood: {like:.4f}\n')
        f.write(f'likelihood gradient: {like_grad}\n')
        if start_time:
            time_curr = time.time() - start_time
            f.write(f'wall time from start: {time_curr:.2f} s \n')

        if self.always_flush:
            f.flush()

    def write_gp_dft_comparison(self, curr_step, frame,
                                start_time, dft_forces,
                                error, local_energies=None, KE=None):
        """ write the comparison to logfile

        :param curr_step: current timestep
        :param frame: Structure object that contain the current GP calculation results
        :param start_time: start time for time profiling
        :param dft_forces: list of forces computed by DFT
        :param error: list of force differences between DFT and GP prediction
        :param local_energies: local atomic energy
        :param KE: total kinetic energy

        :return:
        """

        string = ''

        # Mark if a frame had DFT forces with an asterisk
        string += f"\n*-Frame: {curr_step}"

        # Construct Header line
        string += '\nEl  Position (A) \t\t\t\t '
        string += 'GP Force (ev/A)  \t\t\t\t'
        string += 'Std. Dev (ev/A) \t\t\t\t'
        string += 'DFT Force (ev/A)  \t\t\t\t \n'

        # Construct atom-by-atom description
        for i in range(len(frame.positions)):
            string += f"{frame.species_labels[i]} "
            for j in range(3):
                string += f"{frame.positions[i][j]:10.5} "
            string += '\t'
            for j in range(3):
                string += f"{frame.forces[i][j]:10.5} "
            string += '\t'
            for j in range(3):
                string += f"{frame.stds[i][j]:10.5} "
            string += '\t'
            for j in range(3):
                string += f"{dft_forces[i][j]:10.5} "
            string += '\n'

        string += '\n'

        # self.write_xyz_config(curr_step, frame, forces=frame.forces,
        #                       stds=frame.stds, forces_2=dft_forces,
        #                       dft_step=True)

        mae = np.mean(error) * 1000
        mac = np.mean(np.abs(dft_forces)) * 1000
        string += f'mean absolute error: {mae:.2f} meV/A\n'
        string += f'mean absolute dft component: {mac:.2f} meV/A\n'
        stat = f'{curr_step} {mae:.2} {mac:.2}'

        mae_ps = {}
        count_ps = {}
        species = [Z_to_element(Z) for Z in set(frame.coded_species)]
        for ele in species:
            mae_ps[ele] = 0
            count_ps[ele] = 0
        for atom in range(frame.nat):
            Z = frame.coded_species[atom]
            ele = Z_to_element(Z)
            mae_ps[ele] += np.sum(error[atom, :])
            count_ps[ele] += 1

        string += "mae per species\n"
        for ele in species:
            if count_ps[ele] > 0:
                mae_ps[ele] /= (count_ps[ele] * 3)
                mae_ps[ele] *= 1000  # Put in meV/A
                string += f"type {ele} mae: {mae_ps[ele]:.2f} meV/A\n"
            stat += f' {mae_ps[ele]:.2f}'

        # calculate potential and total energy
        if local_energies is not None:
            pot_en = np.sum(local_energies)
            tot_en = KE + pot_en
            string += f'potential energy: {pot_en:10.6} eV\n'
            string += f'total energy: {tot_en:10.6} eV \n'
            stat += f' {pot_en:10.6} {tot_en:10.6}'

        dt = time.time() - start_time
        string += f'wall time from start: {dt:10.2}\n'
        stat += f' {dt}\n'

        self.outfiles['log'].write(string)
        # self.outfiles['stat'].write(stat)

        if self.always_flush:
            self.outfiles['log'].flush()
