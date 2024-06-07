"""
Class which contains various methods to print the output of different
ways of using FLARE, such as training a GP from an AIMD run,
or running an MD simulation updated on-the-fly.
"""
import datetime
import logging
import time
import sys
import copy
import numpy as np

from logging import FileHandler, StreamHandler, Logger
from os.path import isfile
from shutil import move as movefile
from typing import Union, List

import flare
from ase.data import chemical_symbols
from ase.io import write
from ase.calculators.calculator import PropertyNotImplementedError

# Unit conversions.
eva_to_gpa = 160.21766208


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

    def __init__(
        self,
        basename: str = "otf_run",
        verbose: str = "INFO",
        print_as_xyz: bool = False,
        always_flush: bool = False,
    ):
        """
        Construction. Open files.
        """
        self.basename = f"{basename}"
        self.print_as_xyz = print_as_xyz
        self.always_flush = always_flush

        filesuffix = {"log": ".out", "hyps": "-hyps.dat"}
        if print_as_xyz:
            filesuffix["xyz"] = ".xyz"

        self.logger = []
        for filetype in filesuffix:
            self.open_new_log(filetype, filesuffix[filetype], verbose)

    def conclude_run(self, extra_strings: List[str] = None):
        """
        destruction function that closes all files
        """

        logger = logging.getLogger(self.basename + "log")
        logger.info("-" * 20)
        logger.info("Run complete.")
        if extra_strings is not None:
            for string in extra_strings:
                logger.info(string)
        logging.shutdown()
        self.logger = []

    def open_new_log(self, filetype: str, suffix: str, verbose="info"):
        """
        Open files.  If files with the same
        name are exist, they are backed up with a suffix "-bak".

        :param filetype: the key name for logging
        :param suffix: the suffix of the file to be opened
        :param verbose: the verbose level for the logger
        """

        if filetype not in self.logger:
            set_logger(
                self.basename + filetype,
                stream=False,
                fileout_name=self.basename + suffix,
                verbose=verbose,
            )
            self.logger += [filetype]

    def write_to_log(self, logstring: str, name: str = "log", flush: bool = False):
        """
        Write any string to logfile

        :param logstring: the string to write
        :param name: the key name of the file to logger named 'log'
        :param flush: whether it should be flushed
        """
        logger = logging.getLogger(self.basename + name)
        logger.info(logstring)

        if flush or self.always_flush:
            logger.handlers[0].flush()

    def write_header(
        self,
        gp_str: str,
        dt: float = None,
        Nsteps: int = None,
        structure=None,
        std_tolerance: Union[float, int] = None,
        optional: dict = None,
    ):
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

        f = logging.getLogger(self.basename + "log")
        f.info(f"{datetime.datetime.now()}")

        # Write package version numbers
        f.info(f"python {sys.version}")
        f.info(f"numpy {np.__version__}")
        try:
            import ase

            f.info(f"ase {ase.__version__}")
        except:
            pass

        try:
            import pymatgen

            f.info(f"pymatgen {pymatgen.__version__}")
        except:
            pass

        f.info(f"flare {flare.__version__}")

        # Write uncertainty tolerance
        if isinstance(std_tolerance, tuple):
            std_string = (
                "Relative uncertainty tolerance: "
                f"{std_tolerance[0]} times noise hyperparameter \n"
            )
            std_string += f"Absolute uncertainty tolerance: {std_tolerance[1]} eV/A\n"
        elif std_tolerance < 0:
            std_string = f"Uncertainty tolerance: {np.abs(std_tolerance)} eV/A\n"
        elif std_tolerance > 0:
            std_string = (
                f"Uncertainty tolerance: {np.abs(std_tolerance)} "
                "times noise hyperparameter \n"
            )
        else:
            std_string = ""

        headerstring = "\n"
        headerstring += gp_str
        headerstring += "\n"
        headerstring += std_string
        if dt is not None:
            headerstring += f"Timestep (ps): {dt}\n"
        headerstring += f"Number of frames: {Nsteps}\n"
        if structure is not None:
            headerstring += f"Number of atoms: {structure.nat}\n"
            headerstring += f"System species: {set(structure.symbols)}\n"
            headerstring += "Periodic cell (A): \n"
            headerstring += str(np.array(structure.cell)) + "\n"

        if optional:
            for key, value in optional.items():
                headerstring += f"{key}: {value} \n"

        # report previous positions
        if structure is not None:
            headerstring += "\nPrevious positions (A):\n"
            for i in range(len(structure.positions)):
                headerstring += f"{structure.symbols[i]:5}"
                for j in range(3):
                    headerstring += f"{structure.prev_positions[i][j]:10.4f}"
                headerstring += "\n"
        headerstring += "-" * 80

        f.info(headerstring)

        if self.always_flush:
            f.handlers[0].flush()

    def write_md_header(self, dt, curr_step, dft_step):
        string = ""
        # Mark if a frame had DFT forces with an asterisk
        if not dft_step:
            string += "-" * 80 + "\n"
            string += f"-Frame: {curr_step} "
        else:
            string += f"\n*-Frame: {curr_step} "

        string += f"\nSimulation Time: {(dt * curr_step):10.3f} ps \n"
        return string

    def write_md_config(
        self,
        dt,
        curr_step,
        structure,
        temperature,
        KE,
        start_time,
        dft_step,
        velocities,
    ):
        """write md configuration in log file

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
        string = self.write_md_header(dt, curr_step, dft_step)

        # Construct Header line
        n_space = 30
        string += str.ljust("El", 5)
        string += str.center("Position (A)", n_space)
        string += " " * 4
        if not dft_step:
            string += str.center("GP Force (ev/A)", n_space)
            string += " " * 4
        else:
            string += str.center("DFT Force (ev/A)", n_space)
            string += " " * 4
        string += str.center("Std. Dev (ev/A)", n_space) + " " * 4
        string += str.center("Velocities (A/ps)", n_space) + "\n"

        # Construct atom-by-atom description
        for i in range(len(structure.positions)):
            string += f"{structure.symbols[i]:5}"
            # string += '\t'
            for j in range(3):
                string += f"{structure.positions[i][j]:10.4f}"
            string += " " * 4
            for j in range(3):
                string += f"{structure.forces[i][j]:10.4f}"
            string += " " * 4
            for j in range(3):
                string += f"{structure.stds[i][j]:10.4f}"
            string += " " * 4
            for j in range(3):
                string += f"{velocities[i][j]:10.4f}"
            string += "\n"

        string += "\n"

        # Check if we need to report the stress and pressure tensors
        try:
            if type(structure.stress) == np.ndarray:
                stress_exist = True
        except PropertyNotImplementedError:
            stress_exist = False

        # Report cell if stress attribute is present.
        if stress_exist and structure.stress is not None:
            string += "Periodic cell (A): \n"
            string += str(np.array(structure.cell)) + "\n\n"

        # Report stress tensor.
        pressure = None
        if stress_exist and structure.stress is not None:
            stress_tensor = structure.stress * eva_to_gpa  # Convert to GPa
            s8 = " " * 8
            string += "Stress tensor (GPa):\n"
            string += (
                " " * 7
                + "xx"
                + s8
                + "yy"
                + s8
                + "zz"
                + s8
                + "yz"
                + s8
                + "xz"
                + s8
                + "xy\n"
            )
            for p in range(6):
                string += f"{stress_tensor[p]:10.3f}"
            string += "\n\n"
            pressure = (stress_tensor[0] + stress_tensor[1] + stress_tensor[2]) / 3

        # Report stress tensor uncertainties.
        if stress_exist and structure.stress_stds is not None:
            stress_stds = structure.stress_stds * eva_to_gpa  # Convert to GPa
            string += "Stress tensor uncertainties (GPa):\n"
            for p in range(6):
                string += f"{stress_stds[p]:10.3f}"
            string += "\n\n"

        # Report pressure.
        if pressure is not None:
            string += f"Pressure (GPa): {pressure:.6f} \n"

        string += f"Temperature (K): {temperature:.2f} \n"
        string += f"Kinetic energy (eV): {KE:.6f} \n"

        # Report potential energy.
        if structure.potential_energy is not None:
            string += f"Potential energy (eV): {structure.potential_energy:.6f} \n"

        # Report potential energy uncertainty.
        if structure.local_energy_stds is not None:
            pot_en_std = np.sqrt(np.sum(structure.local_energy_stds**2))
            string += f"Uncertainty (eV): {pot_en_std:.6f} \n"

        # Report total energy.
        if structure.potential_energy is not None:
            tot_en = KE + structure.potential_energy
            string += f"Total energy (eV): {tot_en:.6f} \n"

        logger = logging.getLogger(self.basename + "log")
        logger.info(string)
        self.write_wall_time(start_time)

        if self.always_flush:
            logger.handlers[0].flush()

    def write_xyz_config(
        self,
        curr_step,
        structure,
        forces: np.array = None,
        stds: np.array = None,
        dft_forces: np.array = None,
        dft_energy=0,
        predict_energy=float("nan"),
        target_atoms=None,
    ):
        """write atomic configuration in xyz file

        :param curr_step: Int, number of frames to note in the comment line
        :param structure: FLARE_Atoms, contain positions and forces
        :param forces: Optional list of forces to xyz file
        :param stds: Optional list of uncertanties to xyz file
        :param dft_forces: Optional second list of forces (e.g. DFT forces)

        :return:
        """
        structure.info["target_atoms"] = np.array(target_atoms)
        write(self.basename + ".xyz", structure, append=True)

    def write_hyps(
        self, hyp_labels, hyps, start_time, like, like_grad, name="log", hyps_mask=None
    ):
        """write hyperparameters to logfile

        :param name:
        :param hyp_labels: labels for hyper-parameters. can be None
        :param hyps: list of hyper-parameters
        :param start_time: start time for time profiling
        :param like: likelihood
        :param like_grad: gradient of likelihood

        :return:
        """
        f = logging.getLogger(self.basename + name)

        f.info("\nGP hyperparameters: ")

        if hyp_labels is not None:
            for i, label in enumerate(hyp_labels):
                f.info(f"Hyp{i} : {label:30s} = {hyps[i]:.4f}")
        else:
            for i, hyp in enumerate(hyps):
                f.info(f"Hyp{i} : {hyp:.4f}")
        if isinstance(like, list):
            f.info(f"Likelihood: {np.round(like,4)}")
        else:
            f.info(f"Likelihood: {like:.4f}")
        f.info(f"Likelihood gradient: {like_grad}")

        if start_time:
            self.write_wall_time(start_time)

        if self.always_flush:
            f.handlers[0].flush()

    def write_wall_time(self, start_time, task=None):
        time_curr = time.time() - start_time
        f = logging.getLogger(self.basename + "log")
        if task is None:
            f.info(f"Wall time from start: {time_curr:.2f} s")
        else:
            f.info(f"Time of {task}: {time_curr:.2f} s")

    def conclude_dft(self, dft_count, start_time):
        f = logging.getLogger(self.basename + "log")
        f.info("DFT run complete.")
        f.info(f"Number of DFT calls: {dft_count}")
        self.write_wall_time(start_time)

    def add_atom_info(self, train_atoms, stds):
        f = logging.getLogger(self.basename + "log")
        f.info(f"Adding atom {train_atoms} to the training set.")
        if len(train_atoms) > 0:
            f.info(f"Uncertainty: {stds[train_atoms[0]]}")

    def write_gp_dft_comparison(
        self,
        curr_step,
        frame,
        start_time,
        dft_forces,
        dft_energy,
        error,
        local_energies=None,
        KE=None,
        mgp=False,
        cell=None,
        stress=None,
    ):
        """Write the comparison to logfile.

        :param curr_step: current timestep
        :param frame: FLARE_Atoms object that contains the current GP calculation
            results.
        :param start_time: start time for time profiling
        :param dft_forces: list of forces computed by DFT
        :param dft_energy: total energy computed by DFT
        :param error: list of force differences between DFT and GP prediction
        :param local_energies: local atomic energy
        :param KE: total kinetic energy
        :param cell: print the unit cell of the structure
        :param stress: print the stress acting on the cell

        :return:
        """

        string = ""

        # Mark if a frame had DFT forces with an asterisk
        string += f"\n*-Frame: {curr_step}"

        # Construct Header line
        string += "\nEl  Position (A) \t\t\t\t "
        if mgp:
            string += "M"
        string += "GP Force (ev/A)  \t\t\t\t"
        string += "Std. Dev (ev/A) \t\t\t\t"
        string += "DFT Force (ev/A)  \t\t\t\t \n"

        # Construct atom-by-atom description
        for i in range(len(frame.positions)):
            string += f"{frame.symbols[i]} "
            for j in range(3):
                string += f"{frame.positions[i][j]:10.5} "
            string += "\t"
            for j in range(3):
                string += f"{frame.forces[i][j]:10.5} "
            string += "\t"
            for j in range(3):
                string += f"{frame.stds[i][j]:10.5} "
            string += "\t"
            for j in range(3):
                string += f"{dft_forces[i][j]:10.5} "
            string += "\n"

        string += "\n"

        # Print stress & cell related parameters
        if cell is not None:
            rounded_cell = np.round(cell, 4)
            string += f"cell: {[list(vec) for vec in rounded_cell]} \n"
        if stress:
            raise NotImplementedError

        # Compute errors and errors by species
        mae = np.nanmean(error) * 1000
        mac = np.mean(np.abs(dft_forces)) * 1000
        string += f"mean absolute error: {mae:.2f} meV/A\n"
        string += f"mean absolute dft component: {mac:.2f} meV/A\n"
        stat = f"{curr_step} {mae:.2} {mac:.2}"

        mae_per_species = {}
        count_per_species = {}
        species = [chemical_symbols[Z] for Z in set(frame.numbers)]
        for ele in species:
            mae_per_species[ele] = 0
            count_per_species[ele] = 0

        for atom in range(frame.nat):
            Z = frame.numbers[atom]
            ele = chemical_symbols[Z]
            if np.isnan(np.sum(error[atom, :])):
                continue
            mae_per_species[ele] += np.sum(error[atom, :])
            count_per_species[ele] += 1

        string += "mae per species\n"
        for ele in species:
            if count_per_species[ele] > 0:
                mae_per_species[ele] /= count_per_species[ele] * 3
                mae_per_species[ele] *= 1000  # Put in meV/A
                string += f"type {ele} mae: {mae_per_species[ele]:.2f} meV/A\n"
            stat += f" {mae_per_species[ele]:.2f}"

        # calculate potential and total energy
        if local_energies is not None:
            pot_en = 0
            pot_en = np.sum(local_energies)
            tot_en = KE + pot_en
            string += f"potential energy: {pot_en:10.6} eV (DFT: {dft_energy} eV\n"
            string += f"total energy: {tot_en:10.6} eV \n"
            stat += f" {pot_en:10.6} {tot_en:10.6}"
        else:
            pot_en = float("nan")

        if self.print_as_xyz:
            self.write_xyz_config(
                curr_step,
                frame,
                forces=frame.forces,
                stds=frame.stds,
                dft_forces=dft_forces,
                dft_energy=dft_energy,
                predict_energy=pot_en,
            )

        f = logging.getLogger(self.basename + "log")
        f.info(string)
        self.write_wall_time(start_time)

        # stat += f' {dt}\n'
        # logging.getLogger('stat').write(stat)

        if self.always_flush:
            f.handlers[0].flush()


def add_stream(logger: Logger, verbose: str = "info"):
    """
    set up screen sctream handler to the logger with handlers

    :param logger: the logger
    :param verbose: verbose level
    :type verbose: str
    """

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
    """
    set up file handler to the logger with handlers

    :param logger: the logger
    :param filename: name of the logfile
    :type filename: str
    :param verbose: verbose level
    :type verbose: str
    """

    file_defined = False
    for handler in logger.handlers:
        if isinstance(handler, FileHandler):
            file_defined = True

    if not file_defined:

        # back up
        if isfile(filename):
            movefile(filename, filename + "-bak")

        fh = FileHandler(filename)
        verbose = getattr(logging, verbose.upper())
        logger.setLevel(verbose)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)


def set_logger(
    name: str, stream: bool, fileout_name: str = None, verbose: str = "info"
):
    """
    set up a logger with handlers

    :param name: unique name of the logger in logging module
    :type name: str
    :param stream: if True, set up a screen output
    :type stream: bool
    :param fileout_name: name for log file
    :type fileout_name: str
    :param verbose: verbose level
    :type verbose: str
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.handlers = []
    logger.setLevel(getattr(logging, verbose.upper()))
    if stream:
        add_stream(logger, verbose)
    if fileout_name is not None:
        add_file(logger, fileout_name, verbose)
    return logger


def compute_mae(
    atoms,
    output_name,
    gp_energy,
    gp_forces,
    gp_stress,
    dft_energy,
    dft_forces,
    dft_stress,
    force_only=False,
):

    f = logging.getLogger(output_name + "log")
    f.info("Mean absolute errors & Mean absolute values")

    # compute energy/forces/stress mean absolute error and value
    e_mae = e_mav = s_mae = s_mav = 0
    if not force_only:
        if dft_energy is not None and gp_energy is not None:
            e_mae = np.mean(np.abs(dft_energy - gp_energy))
            e_mav = np.mean(np.abs(dft_energy))
            f.info(f"energy mae: {e_mae:.4f} eV")
            f.info(f"energy mav: {e_mav:.4f} eV")

        if dft_stress is not None and gp_stress is not None:
            s_mae = np.mean(np.abs(dft_stress - gp_stress))
            s_mav = np.mean(np.abs(dft_stress))
            f.info(f"stress mae: {s_mae:.4f} eV/A^3")
            f.info(f"stress mav: {s_mav:.4f} eV/A^3")

    f_mae = np.mean(np.abs(dft_forces - gp_forces))
    f_mav = np.mean(np.abs(dft_forces))
    f.info(f"forces mae: {f_mae:.4f} eV/A")
    f.info(f"forces mav: {f_mav:.4f} eV/A")

    # compute the per-species MAE
    unique_species = list(set(atoms.numbers))
    per_species_mae = np.zeros(len(unique_species))
    per_species_mav = np.zeros(len(unique_species))
    per_species_num = np.zeros(len(unique_species))
    for a in range(atoms.nat):
        species_ind = unique_species.index(atoms.numbers[a])
        per_species_mae[species_ind] += np.mean(np.abs(dft_forces[a] - gp_forces[a]))
        per_species_mav[species_ind] += np.mean(np.abs(dft_forces[a]))
        per_species_num[species_ind] += 1
    per_species_mae /= per_species_num
    per_species_mav /= per_species_num

    for s in range(len(unique_species)):
        curr_species = unique_species[s]
        f.info(f"type {curr_species} forces mae: {per_species_mae[s]:.4f} eV/A")
        f.info(f"type {curr_species} forces mav: {per_species_mav[s]:.4f} eV/A")

    return e_mae, e_mav, f_mae, f_mav, s_mae, s_mav
