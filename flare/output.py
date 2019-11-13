import datetime
import os
import shutil
import time

import multiprocessing
import numpy as np

from flare.util import Z_to_element


class Output():
    """
    Output class host the logfile, xyz config file and force xyz file.
    :param basename
    """

    def __init__(self, basename: str = 'otf_run',
                 always_flush: bool = False):
        """
        Ppen files with the basename and different suffixes corresponding
        to different kinds of output data.

        :param basename: Base output file name, suffixes will be added
        :param always_flush: Always write to file instantly
        """
        self.basename = f"{basename}"
        self.outfiles = {}
        filesuffix = {'log': '.out', 'xyz': '.xyz',
                      'fxyz': '-f.xyz', 'hyps': "-hyps.dat",
                      'std': '-std.xyz', 'stat': '-stat.dat'}

        for filetype in filesuffix.keys():
            self.open_new_log(filetype, filesuffix[filetype])

        self.always_flush = always_flush

    def conclude_run(self):
        """
        destruction function that close all files
        """
        print('-' * 20, file=self.outfiles['log'])
        print('Run complete.', file=self.outfiles['log'])
        for (k, v) in self.outfiles.items():
            v.close()
        del self.outfiles
        self.outfiles = {}

    def open_new_log(self, filetype, suffix):

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
        """
        self.outfiles[name].write(logstring)

        if flush or self.always_flush:
            self.outfiles[name].flush()

    def write_header(self, cutoffs, kernel_name,
                     hyps, algo, dt, Nsteps, structure,
                     std_tolerance,
                     optional: dict = None):
        """
        write header to the log function
        :param cutoffs:
        :param kernel_name:
        :param hyps:
        :param algo:
        :param dt:
        :param Nsteps:
        :param structure:
        :param std_tolerance:
        :param optional:
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
            headerstring += f'{structure.species_labels[i]} '
            for j in range(3):
                headerstring += f'{structure.prev_positions[i][j]:10.4} '
            headerstring += '\n'
        headerstring += '-' * 80 + '\n'

        f.write(headerstring)

        if self.always_flush:
            f.flush()

    # TO DO: this module should be removed in the future
    def write_md_config(self, dt, curr_step, structure,
                        temperature, KE, local_energies,
                        start_time, dft_step, velocities):
        """
        write md configuration in log file
        :param dt:
        :param curr_step:
        :param structure:
        :param temperature:
        :param KE:
        :param local_energies:
        :param start_time:
        :param dft_step:
        :param velocities:
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
        string += 'El  Position (A) \t\t\t\t '
        if not dft_step:
            string += 'GP Force (ev/A) '
        else:
            string += 'DFT Force (ev/A) '
        string += '\t\t\t\t Std. Dev (ev/A) \t'
        string += '\t\t\t\t Velocities (A/ps) \n'

        # Construct atom-by-atom description
        for i in range(len(structure.positions)):
            string += f'{structure.species_labels[i]} '
            for j in range(3):
                string += f"{structure.positions[i][j]:.8}"
            string += '\t'
            for j in range(3):
                string += f"{structure.forces[i][j]:8.3} "
            string += '\t'
            for j in range(3):
                string += f'{structure.stds[i][j]:.8e}'
            string += '\t'
            for j in range(3):
                string += f'{velocities[i][j]:.8e}'
            string += '\n'

        print(curr_step)
        print(structure.species_labels)
        self.write_xyz_config(curr_step, structure, dft_step)
        self.write_xyz(curr_step, structure.stds, structure.species_labels,
                       "std", header)

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
                  forces: np.array=None, stds: np.array=None,
                  forces_2: np.array=None):
        """
        write atomic configuration in xyz file
        :param curr_step: Int, number of frames to note in the comment line
        :param pos:       nx3 matrix of forces, positions, or nything
        :param species:   n element list of symbols
        :param filename:  file to print
        :param header:    header printed in comments
        :param forces: list of forces on atoms predicted by GP
        :param stds: uncertainties predicted by GP
        :param forces_2: true forces from ab initio source
        :return:
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
                         forces: np.array = None, stds : np.array = None,
                         forces_2: np.array = None):
        """
        write atomic configuration in xyz file
        :param curr_step: Int, number of frames to note in the comment line
        :param structure: Structure, contain positions and forces
        :param dft_step:  Boolean, whether this is a DFT call.
        :param forces: Optional list of forces to print in xyz file
        :param stds: Optional list of uncertanties to print in xyz file
        :param forces_2: Optional second list of forces (e.g. DFT forces)
        :return:
        """

        # comment line
        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            header = ""
        else:
            header = "*"
        self.write_xyz(curr_step=curr_step, pos=structure.positions,
                       species=structure.species_labels, filename='xyz',
                       header=header,
                       forces=forces, stds = stds, forces_2=forces_2)

    def write_hyps(self, hyp_labels, hyps, start_time, like, like_grad,
                   name='log'):
        """
        write hyperparameters to logfile
        :param hyp_labels:
        :param hyps:
        :param start_time:
        :param like:
        :param like_grad:
        :return:
        """
        f = self.outfiles[name]
        f.write('\nGP hyperparameters: \n')

        if (hyp_labels is not None):
            for i, label in enumerate(hyp_labels):
                f.write(f'Hyp{i} : {label} = {hyps[i]}\n')
        else:
            for i, hyp in enumerate(hyps):
                f.write(f'Hyp{i} : {hyp}\n')

        f.write(f'likelihood: {like}\n')
        f.write(f'likelihood gradient: {like_grad}\n')
        if (start_time):
            time_curr = time.time() - start_time
            f.write(f'wall time from start: {time_curr:.2} s \n')

        if self.always_flush:
            f.flush()

    def write_gp_dft_comparison(self, curr_step, frame,
                                start_time, dft_forces,
                                error, local_energies=None, KE=None):
        """
        write the comparison to logfile
        :param dft_forces:
        :param mae:
        :param pmae: dictionary of per species mae
        :param mac:
        :param KE:
        :param curr_step:
        :param frame:
        :param local_energies:
        :param start_time:
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
                string += f"{frame.positions[i][j]:10.3} "
            string += '\t'
            for j in range(3):
                string += f"{frame.forces[i][j]:10.3} "
            string += '\t'
            for j in range(3):
                string += f"{frame.stds[i][j]:10.3} "
            string += '\t'
            for j in range(3):
                string += f"{dft_forces[i][j]:10.3} "
            string += '\n'

        string += '\n'

        self.write_xyz_config(curr_step, frame, forces = frame.forces,
                              stds = frame.stds, forces_2 = dft_forces,
                              dft_step=True)

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
            if (count_ps[ele] > 0):
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
        self.outfiles['stat'].write(stat)

        if self.always_flush:
            self.outfiles['log'].flush()
