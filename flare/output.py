"""
Class which contains various methods to print the output of different
ways of using FLARE, such as training a GP from an AIMD run,
or running an MD simulation updated on-the-fly.
"""
import datetime
import logging
import time
import numpy as np

from logging import FileHandler, StreamHandler, Logger
from os.path import isfile
from shutil import move as movefile
from typing import Union

from flare.struc import Structure
from flare.utils.element_coder import Z_to_element


class Output:
    """
    This is an I/O class that hosts the log files for OTF and Trajectories
    class. It is also used in get_neg_like_grad and get_neg_likelihood in
    gp_algebra to print intermediate results.

    It opens and print files with the basename prefix and different
    suffixes corresponding to different kinds of output data.

    :param basename: Base output file name, suffixes will be added
    :type basename: str, optional
    :param verbose: print level. The same as logging level. It can be
                    CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    :type verbose: str, optional
    :param always_flush: Always write to file instantly
    :type always_flus: bool, optional
    """

    def __init__(self, basename: str = 'otf_run',
                 verbose: str = 'INFO',
                 always_flush: bool = False):
        """
        Construction. Open files.
        """
        self.basename = f"{basename}"
        filesuffix = {'log': '.out', 'hyps': '-hyps.dat'}
        self.logger = []

        for filetype in filesuffix:
            self.open_new_log(filetype, filesuffix[filetype], verbose)

        self.always_flush = always_flush

    def conclude_run(self):
        """
        destruction function that closes all files
        """

        logger = logging.getLogger(self.basename+'log')
        logger.info('-' * 20)
        logger.info('Run complete.')
        logging.shutdown()
        self.logger = []

    def open_new_log(self, filetype: str, suffix: str, verbose='info'):
        """
        Open files.  If files with the same
        name are exist, they are backed up with a suffix "-bak".

        :param filetype: the key name for logging
        :param suffix: the suffix of the file to be opened
        :param verbose: the verbose level for the logger
        """

        if filetype not in self.logger:
            set_logger(self.basename+filetype, stream=False,
                       fileout_name=self.basename+suffix,
                       verbose=verbose)
            self.logger += [filetype]

    def write_to_log(self, logstring: str, name: str = "log",
                     flush: bool = False):
        """
        Write any string to logfile

        :param logstring: the string to write
        :param name: the key name of the file to logger named 'log'
        :param flush: whether it should be flushed
        """
        logger = logging.getLogger(self.basename+name)
        logger.info(logstring)

        if flush or self.always_flush:
            logger.handlers[0].flush()

    def write_header(self, gp_str: str,
                     dt: float = None,
                     Nsteps: int = None, structure: Structure = None,
                     std_tolerance: Union[float, int] = None,
                     optional: dict = None):
        """
        TO DO: this should be replace by the string method of GP and OTF, GPFA

        Write header to the log function. Designed for Trajectory Trainer and
        OTF runs and can take flexible input for both.

        :param gp_str: string representation of the GP
        :param dt: timestep for OTF MD
        :param Nsteps: total number of steps for OTF MD
        :param structure: initial structure
        :param std_tolerance: tolarence for active learning
        :param optional: a dictionary of all the other parameters
        """

        f = logging.getLogger(self.basename+'log')
        f.info(f'{datetime.datetime.now()}')

        if isinstance(std_tolerance, tuple):
            std_string = 'relative uncertainty tolerance: ' \
                         f'{std_tolerance[0]} times noise hyperparameter \n'
            std_string += 'absolute uncertainty tolerance: ' \
                          f'{std_tolerance[1]} eV/A\n'
        elif std_tolerance < 0:
            std_string = \
                f'uncertainty tolerance: {np.abs(std_tolerance)} eV/A\n'
        elif std_tolerance > 0:
            std_string = \
                f'uncertainty tolerance: {np.abs(std_tolerance)} ' \
                'times noise hyperparameter \n'
        else:
            std_string = ''

        headerstring = '\n'
        headerstring += gp_str
        headerstring += '\n'
        headerstring += std_string
        if dt is not None:
            headerstring += f'timestep (ps): {dt}\n'
        headerstring += f'number of frames: {Nsteps}\n'
        if structure is not None:
            headerstring += f'number of atoms: {structure.nat}\n'
            headerstring += f'system species: {set(structure.species_labels)}\n'
            headerstring += 'periodic cell: \n'
            headerstring += str(structure.cell)+'\n'

        if optional:
            for key, value in optional.items():
                headerstring += f"{key}: {value} \n"

        # report previous positions
        if structure is not None:
            headerstring += '\nprevious positions (A):\n'
            for i in range(len(structure.positions)):
                headerstring += f'{structure.species_labels[i]:5}'
                for j in range(3):
                    headerstring += f'{structure.prev_positions[i][j]:10.4f}'
                headerstring += '\n'
        headerstring += '-' * 80 + '\n'

        f.info(headerstring)

        if self.always_flush:
            f.handlers[0].flush()

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
        tab = ' ' * 4

        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += '-' * 80 + '\n'
            string += f"-Frame: {curr_step} "
        else:
            string += f"\n*-Frame: {curr_step} "

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

        logger = logging.getLogger(self.basename+'log')
        logger.info(string)
        self.write_wall_time(start_time)

        if self.always_flush:
            logger.handlers[0].flush()

    def write_xyz(self, curr_step: int, pos: np.array, species: list,
                  filename: str,
                  header="",
                  forces: np.array = None, stds: np.array = None,
                  forces_2: np.array = None):
        """ write atomic configuration in xyz file

        :param curr_step: Int, number of frames to note in the comment line
        :param pos:       nx3 matrix of forces, positions, or nything
        :param species:   n element list of symbols
        :param filename:  key of logger
        :param header:    header in comments
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
        logger = logging.getLogger(self.basename+filename)
        logger.info(string)

        if self.always_flush:
            logger.handlers[0].flush()

    def write_xyz_config(self, curr_step, structure, dft_step,
                         forces: np.array = None, stds: np.array = None,
                         forces_2: np.array = None):
        """ write atomic configuration in xyz file

        :param curr_step: Int, number of frames to note in the comment line
        :param structure: Structure, contain positions and forces
        :param dft_step:  Boolean, whether this is a DFT call.
        :param forces: Optional list of forces to xyz file
        :param stds: Optional list of uncertanties to xyz file
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
                   name='log', hyps_mask=None):
        """ write hyperparameters to logfile

        :param name:
        :param hyp_labels: labels for hyper-parameters. can be None
        :param hyps: list of hyper-parameters
        :param start_time: start time for time profiling
        :param like: likelihood
        :param like_grad: gradient of likelihood

        :return:
        """
        f = logging.getLogger(self.basename+name)

        f.info('\nGP hyperparameters: ')

        if hyp_labels is not None:
            for i, label in enumerate(hyp_labels):
                f.info(f'Hyp{i} : {label} = {hyps[i]:.4f}')
        else:
            for i, hyp in enumerate(hyps):
                f.info(f'Hyp{i} : {hyp:.4f}')

        f.info(f'likelihood: {like:.4f}')
        f.info(f'likelihood gradient: {like_grad}')

        if start_time:
            self.write_wall_time(start_time)

        if self.always_flush:
            f.handlers[0].flush()

    def write_wall_time(self, start_time):
        time_curr = time.time() - start_time
        f = logging.getLogger(self.basename+'log')
        f.info(f'wall time from start: {time_curr:.2f} s')

    def conclude_dft(self, dft_count, start_time):
        f = logging.getLogger(self.basename+'log')
        f.info('DFT run complete.')
        f.info(f'number of DFT calls: {dft_count}')
        self.write_wall_time(start_time)

    def add_atom_info(self, train_atoms, stds):
        f = logging.getLogger(self.basename+'log')
        f.info(f'Adding atom {train_atoms} to the training set.')
        f.info(f'Uncertainty: {stds[train_atoms[0]]}')

    def write_gp_dft_comparison(self, curr_step, frame,
                                start_time, dft_forces,
                                error, local_energies=None, KE=None,
                                mgp=False):
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
        if mgp:
            string += 'M'
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

        mae = np.nanmean(error) * 1000
        mac = np.mean(np.abs(dft_forces)) * 1000
        string += f'mean absolute error: {mae:.2f} meV/A\n'
        string += f'mean absolute dft component: {mac:.2f} meV/A\n'
        stat = f'{curr_step} {mae:.2} {mac:.2}'

        mae_per_species = {}
        count_per_species = {}
        species = [Z_to_element(Z) for Z in set(frame.coded_species)]
        for ele in species:
            mae_per_species[ele] = 0
            count_per_species[ele] = 0

        for atom in range(frame.nat):
            Z = frame.coded_species[atom]
            ele = Z_to_element(Z)
            if np.isnan(np.sum(error[atom, :])):
                continue
            mae_per_species[ele] += np.sum(error[atom, :])
            count_per_species[ele] += 1

        string += "mae per species\n"
        for ele in species:
            if count_per_species[ele] > 0:
                mae_per_species[ele] /= (count_per_species[ele] * 3)
                mae_per_species[ele] *= 1000  # Put in meV/A
                string += f"type {ele} mae: {mae_per_species[ele]:.2f} meV/A\n"
            stat += f' {mae_per_species[ele]:.2f}'

        # calculate potential and total energy
        if local_energies is not None:
            pot_en = np.sum(local_energies)
            tot_en = KE + pot_en
            string += f'potential energy: {pot_en:10.6} eV\n'
            string += f'total energy: {tot_en:10.6} eV \n'
            stat += f' {pot_en:10.6} {tot_en:10.6}'

        f = logging.getLogger(self.basename+'log')
        f.info(string)
        self.write_wall_time(start_time)

        # stat += f' {dt}\n'
        # logging.getLogger('stat').write(stat)

        if self.always_flush:
            f.handlers[0].flush()


def add_stream(logger: Logger, verbose: str = "info"):
    '''
    set up screen sctream handler to the logger with handlers

    :param logger: the logger
    :param verbose: verbose level
    :type verbose: str
    '''

    stream_defined = False
    for handler in logger.handlers:
        if isinstance(handler, StreamHandler):
            stream_defined = True

    if not stream_defined:
        ch = StreamHandler()
        ch.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # ch.setFormatter(formatter)
        logger.addHandler(ch)


def add_file(logger: Logger, filename: str, verbose: str = "info"):
    '''
    set up file handler to the logger with handlers

    :param logger: the logger
    :param filename: name of the logfile
    :type filename: str
    :param verbose: verbose level
    :type verbose: str
    '''

    file_defined = False
    for handler in logger.handlers:
        if isinstance(handler, FileHandler):
            file_defined = True

    if not file_defined:

        # back up
        if isfile(filename):
            movefile(filename, filename+"-bak")

        fh = FileHandler(filename)
        verbose = getattr(logging, verbose.upper())
        logger.setLevel(verbose)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)


def set_logger(name: str, stream: bool, fileout_name: str = None,
               verbose: str = "info"):
    '''
    set up a logger with handlers

    :param name: unique name of the logger in logging module
    :type name: str
    :param stream: if True, set up a screen output
    :type stream: bool
    :param fileout_name: name for log file
    :type fileout_name: str
    :param verbose: verbose level
    :type verbose: str
    '''
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.handlers = []
    logger.setLevel(getattr(logging, verbose.upper()))
    if stream:
        add_stream(logger, verbose)
    if fileout_name is not None:
        add_file(logger, fileout_name, verbose)
    return logger
