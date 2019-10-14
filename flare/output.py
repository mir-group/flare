import time
import datetime
import numpy as np
import multiprocessing
import os
import shutil


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
                      'fxyz': '-f.xyz', 'hyps': "-hyps.dat"}

        for filetype in filesuffix.keys():
            self.open_new_log(filetype, filesuffix[filetype])

        self.always_flush = always_flush

    def conclude_run(self):
        """
        destruction function that close all files
        """
        print('-'*20, file=self.outfiles['log'])
        print('Run complete.', file=self.outfiles['log'])
        for (k, v) in self.outfiles.items():
            v.close()
        del self.outfiles
        self.outfiles = {}

    def open_new_log(self, filetype, suffix):

        filename = self.basename+suffix

        # if the file exists, back up
        if os.path.isfile(filename):
            shutil.copy(filename, filename+"-bak")

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
        f.write(str(datetime.datetime.now()) + '\n')

        if isinstance(std_tolerance, tuple):
            std_string = 'relative uncertainty tolerance: '\
                    f'{std_tolerance[0]} eV/A\n'
            std_string += 'absolute uncertainty tolerance: '\
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
            string += "-Frame: " + str(curr_step)
        else:
            string += "\n*-Frame: " + str(curr_step)

        string += '\nSimulation Time: %.3f ps \n' % (dt * curr_step)

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
            string += str(structure.species_labels[i]) + ' '
            for j in range(3):
                string += str("%.8f" % structure.positions[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8f" % structure.forces[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8e" % structure.stds[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8e" % velocities[i][j]) + ' '
            string += '\n'

        string += '\n'
        string += 'temperature: %.2f K \n' % temperature
        string += 'kinetic energy: %.6f eV \n' % KE

        # calculate potential and total energy
        if local_energies is not None:
            pot_en = np.sum(local_energies)
            tot_en = KE + pot_en
            string += \
                'potential energy: %.6f eV \n' % pot_en
            string += 'total energy: %.6f eV \n' % tot_en

        string += 'wall time from start: %.2f s \n' % \
            (time.time() - start_time)

        self.outfiles['log'].write(string)

        if self.always_flush:
            self.outfiles['log'].flush()

    def write_xyz_config(self, curr_step, structure, dft_step):
        """
        write atomic configuration in xyz file
        :param curr_step: Int, number of frames to note in the comment line
        :param structure: Structure, contain positions and forces
        :param dft_step:  Boolean, whether this is a DFT call.
        :return:
        """

        natom = len(structure.positions)
        string = f'{natom}\n'

        # comment line
        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += f"Frame: {curr_step}\n"
        else:
            string += f"*Frame: {curr_step}\n"

        # Construct atom-by-atom description
        for i in range(natom):
            pos = structure.positions[i]
            string += f'{structure.species_labels[i]} '
            string += f'{pos[0]} {pos[1]} {pos[2]}\n'

        self.outfiles['xyz'].write(string)

        string = f'{natom}\n'

        # comment line
        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += f"Frame: {curr_step}\n"
        else:
            string += f"*Frame: {curr_step}\n"

        # Construct atom-by-atom description
        for i in range(natom):
            pos = structure.forces[i]
            string += f'{structure.species_labels[i]} '
            string += f'{pos[0]} {pos[1]} {pos[2]}\n'

        self.outfiles['fxyz'].write(string)

        if self.always_flush:
            self.outfiles['xyz'].flush()
            self.outfiles['fxyz'].flush()

    def write_hyps(self, hyp_labels, hyps, start_time, like, like_grad, name='log'):
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

        f.write('likelihood: {like}\n')
        f.write('likelihood gradient: {like_grad}\n')
        if (start_time):
            time_curr = time.time() - start_time
            f.write('wall time from start: {time_curr:.2} s \n')

        if self.always_flush:
            f.flush()

    def write_gp_dft_comparison(self, curr_step, frame,
                                start_time, dft_forces,
                                mae, mae_ps, mac, local_energies=None, KE=None):
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
            for j in range(3)                            :
                string += f"{frame.forces[i][j]:10.3} "
            string += '\t'
            for j in range(3):
                string += f"{frame.stds[i][j]:10.3} "
            string += '\t'
            for j in range(3):
                string += f"{dft_forces[i][j]:10.3} "
            string += '\n'

        string += '\n'

        string += f'mean absolute error: {mae:10.2} meV/A \n'
        string += f'mean absolute dft component: {mac:10.2} meV/A \n'

        string += "mae per species\n"
        for ele in mae_ps.keys():
            string+=f"type {ele} mae: {mae_ps[ele]:10.4}\n"

        # calculate potential and total energy
        if local_energies is not None:
            pot_en = np.sum(local_energies)
            tot_en = KE + pot_en
            string += f'potential energy: {pot_en:10.6} eV\n'
            string += f'total energy: {tot_en:10.6} eV \n'

        string += f'wall time from start: {time.time() - start_time:10.2}\n'

        self.outfiles['log'].write(string)

        if self.always_flush:
            self.outfiles['log'].flush()
