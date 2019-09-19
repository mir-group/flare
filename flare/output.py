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

    def __init__(self, basename: str = 'otf_run'):
        """
        open files with the basename and different suffices.
        """
        self.basename = "{}".format(basename)
        self.outfiles = {}
        filesuffix = {'log': '.out', 'xyz': '.xyz', 'fxyz': '-f.xyz', 'hyps':"-hyps.dat"}

        for filetype in filesuffix.keys():
            self.open_new_log(filetype, filesuffix[filetype])

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

    def write_to_log(self, logstring: str, name: str = "log"):
        """
        Write any string to logfile
        """
        self.outfiles[name].write(logstring)

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
            std_string = "relative uncertainty tolerance: {} eV/A\n".format(
                std_tolerance[0])
            std_string += 'absolute uncertainty tolerance: {} eV/A\n'.format(
                std_tolerance[1])
        elif std_tolerance < 0:
            std_string = \
                'uncertainty tolerance: {} eV/A\n'.format(
                    np.abs(std_tolerance))
        elif std_tolerance > 0:
            std_string = \
                'uncertainty tolerance: {} times noise \n' \
                .format(np.abs(std_tolerance))
        else:
            std_string = ''

        headerstring = ''
        headerstring += \
            'number of cpu cores: {}\n'.format(multiprocessing.cpu_count())
        headerstring += 'cutoffs: {}\n'.format(cutoffs)
        headerstring += 'kernel: {}\n'.format(kernel_name)
        headerstring += 'number of hyperparameters: {}\n'.format(len(hyps))
        headerstring += 'hyperparameters: {}' \
                        '\n'.format(hyps)
        headerstring += 'hyperparameter optimization algorithm: {}' \
                        '\n'.format(algo)
        headerstring += std_string
        headerstring += 'timestep (ps): {}\n'.format(dt)
        headerstring += 'number of frames: {}\n'.format(Nsteps)
        headerstring += 'number of atoms: {}\n'.format(structure.nat)
        headerstring += \
            'system species: {}\n'.format(set(structure.species_labels))
        headerstring += 'periodic cell: \n'
        headerstring += str(structure.cell)

        if optional:
            for key, value in optional.items():
                headerstring += "{}: {} \n".format(key, value)

        # report previous positions
        headerstring += '\nprevious positions (A):\n'
        for i in range(len(structure.positions)):
            headerstring += str(structure.species_labels[i]) + ' '
            for j in range(3):
                headerstring += str("%.8f" %
                                    structure.prev_positions[i][j]) + ' '
            headerstring += '\n'
        headerstring += '-' * 80 + '\n'

        f.write(headerstring)

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

    def write_xyz_config(self, curr_step, structure, dft_step):
        """
        write atomic configuration in xyz file
        :param curr_step: Int, number of frames to note in the comment line
        :param structure: Structure, contain positions and forces
        :param dft_step:  Boolean, whether this is a DFT call.
        :return:
        """

        natom = len(structure.positions)
        string = '{}\n'.format(natom)

        # comment line
        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += "Frame: {}\n".format(curr_step)
        else:
            string += "*Frame: {}\n".format(curr_step)

        # Construct atom-by-atom description
        for i in range(natom):
            pos = structure.positions[i]
            string += '{} {} {} {}\n'.format(
                structure.species_labels[i], pos[0], pos[1], pos[2])

        self.outfiles['xyz'].write(string)

        string = '{}\n'.format(natom)

        # comment line
        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += "Frame: {}\n".format(curr_step)
        else:
            string += "*Frame: {}\n".format(curr_step)

        # Construct atom-by-atom description
        for i in range(natom):
            pos = structure.forces[i]
            string += '{} {} {} {}\n'.format(
                structure.species_labels[i], pos[0], pos[1], pos[2])

        self.outfiles['fxyz'].write(string)

    def write_hyps(self, hyp_labels, hyps, start_time, like, like_grad):
        """
        write hyperparameters to logfile
        :param hyp_labels:
        :param hyps:
        :param start_time:
        :param like:
        :param like_grad:
        :return:
        """
        f = self.outfiles['log']
        f.write('\nGP hyperparameters: \n')

        for i, label in enumerate(hyp_labels):
            f.write('Hyp{} : {} = {}\n'.format(i, label, hyps[i]))

        f.write('likelihood: ' + str(like) + '\n')
        f.write('likelihood gradient: ' + str(like_grad) + '\n')
        time_curr = time.time() - start_time
        f.write('wall time from start: %.2f s \n' % time_curr)

    def write_gp_dft_comparison(self, curr_step, frame,
                                start_time, dft_forces,
                                mae, mac, local_energies=None, KE=None):
        """
        write the comparison to logfile
        :param dft_forces:
        :param mae:
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
        string += "\n*-Frame: " + str(curr_step)

        # Construct Header line
        string += '\nEl  Position (A) \t\t\t\t '
        string += 'GP Force (ev/A)  \t\t\t\t'
        string += 'Std. Dev (ev/A) \t\t\t\t'
        string += 'DFT Force (ev/A)  \t\t\t\t \n'

        # Construct atom-by-atom description
        for i in range(len(frame.positions)):
            string += str(frame.species_labels[i]) + ' '
            for j in range(3):
                string += str("%.8f" % frame.positions[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8f" % frame.forces[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8e" % frame.stds[i][j]) + ' '
            string += '\t'
            for j in range(3):
                string += str("%.8f" % dft_forces[i][j]) + ' '
            string += '\n'

        string += '\n'

        string += 'mean absolute error: %.2f meV/A \n' % mae
        string += 'mean absolute dft component: %.2f meV/A \n' % mac

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
