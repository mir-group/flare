import sys
import numpy as np
from typing import List, Tuple
sys.path.append('../../otf/otf_engine')
import gp, env, struc, kernels, otf


class OtfAnalysis:
    def __init__(self, filename, calculate_energy=False):
        self.filename = filename

        self.calculate_energy = calculate_energy

        position_list, force_list, uncertainty_list, velocity_list,\
            dft_frames, temperatures, times, msds, dft_times, energies = \
            self.parse_pos_otf(filename)

        self.position_list = position_list
        self.force_list = force_list
        self.uncertainty_list = uncertainty_list
        self.velocity_list = velocity_list
        self.dft_frames = dft_frames
        self.temperatures = temperatures
        self.times = times
        self.msds = msds
        self.dft_times = dft_times

        if self.calculate_energy:
            self.energies = energies

        gp_position_list, gp_force_list, gp_uncertainty_list,\
            gp_velocity_list, gp_atom_list, gp_hyp_list, \
            gp_species_list, gp_atom_count = self.extract_gp_info(filename)

        self.gp_position_list = gp_position_list
        self.gp_force_list = gp_force_list
        self.gp_uncertainty_list = gp_uncertainty_list
        self.gp_velocity_list = gp_velocity_list
        self.gp_atom_list = gp_atom_list
        self.gp_hyp_list = gp_hyp_list
        self.gp_species_list = gp_species_list
        self.gp_atom_count = gp_atom_count

    def make_gp(self, cell, kernel, kernel_grad, algo, call_no, cutoffs):
        gp_hyps = self.gp_hyp_list[call_no-1][-1]
        gp_model = gp.GaussianProcess(kernel, kernel_grad, gp_hyps,
                                      cutoffs, opt_algorithm=algo)

        for (positions, forces, atoms, _, species) in \
            zip(self.gp_position_list, self.gp_force_list,
                self.gp_atom_list, self.gp_hyp_list,
                self.gp_species_list):

            struc_curr = struc.Structure(cell, species, positions)

            gp_model.update_db(struc_curr, forces, custom_range=atoms)

        gp_model.set_L_alpha()

        return gp_model

    @staticmethod
    def get_gp_activation(gp_model):
        pass

    def parse_pos_otf(self, filename):
        """
        Exclusively parses MD run information
        :param filename:
        :return:
        """
        position_list = []
        force_list = []
        uncertainty_list = []
        velocity_list = []
        temperatures = []
        dft_frames = []
        dft_times = []
        times = []
        msds = []
        energies = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        n_steps = 0

        for index, line in enumerate(lines):
            if line.startswith("number of atoms"):
                at_line = line.split()
                noa = int(at_line[3])

            # number of hyperparameters
            if line.startswith("number of hyperparameters"):
                line_curr = line.split(':')
                noh = int(line_curr[-1])

            # DFT frame
            if line.startswith("*-Frame"):
                dft_frame_line = line.split()
                dft_frames.append(int(dft_frame_line[1]))
                dft_time_line = lines[index+1].split()
                dft_times.append(float(dft_time_line[-2]))

            # MD frame
            if line.startswith("-Frame"):
                n_steps += 1
                time_line = lines[index+1].split()
                sim_time = float(time_line[2])
                times.append(sim_time)

                _, positions, forces, uncertainties, velocities = \
                    parse_snapshot(lines, index, noa, False, noh)

                position_list.append(positions)
                force_list.append(forces)
                uncertainty_list.append(uncertainties)
                velocity_list.append(velocities)

                temp_line = lines[index+4+noa].split()
                temperatures.append(float(temp_line[-2]))

                if self.calculate_energy:
                    en_line = lines[index+5+noa].split()
                    energies.append(float(en_line[2]))

                msds.append(np.mean((positions - position_list[0])**2))

        return position_list, force_list, uncertainty_list, velocity_list,\
            dft_frames, temperatures, times, msds, dft_times, energies

    def extract_gp_info(self, filename):
        """
        Exclusively parses DFT run information
        :param filename:
        :return:
        """
        species_list = []
        position_list = []
        force_list = []
        uncertainty_list = []
        velocity_list = []
        atom_list = []
        atom_count = []
        hyp_list = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        for index, line in enumerate(lines):

            # number of atoms
            if line.startswith("number of atoms"):
                line_curr = line.split()
                noa = int(line_curr[3])

            # number of hyperparameters
            if line.startswith("number of hyperparameters"):
                line_curr = line.split()
                noh = int(line_curr[3])

            if line.startswith("*-"):
                # keep track of atoms added to training set
                ind_count = index+1
                line_check = lines[ind_count]
                atoms_added = []
                hyps_added = []
                while not line_check.startswith("-"):
                    # keep track of atom number
                    if line_check.startswith("Adding atom"):
                        line_split = line_check.split()
                        atoms_added.append(int(line_split[2]))

                    # keep track of hyperparameters
                    if line_check.startswith("GP hyperparameters:"):
                        hyps = []
                        for hyp_line in lines[(ind_count+1):(ind_count+1+noh)]:
                            hyp_line = hyp_line.split()
                            hyps.append(float(hyp_line[-1]))
                        hyps = np.array(hyps)
                        hyps_added.append(hyps)

                    ind_count += 1
                    line_check = lines[ind_count]

                hyp_list.append(hyps_added)
                atom_list.append(atoms_added)
                atom_count.append(len(atoms_added))

                # add DFT positions and forces
                line_curr = line.split()

                # TODO: generalize this to account for arbitrary starting list
                append_atom_lists(species_list, position_list, force_list,
                                  uncertainty_list, velocity_list,
                                  lines, index, noa, True, noh)

        return position_list, force_list, uncertainty_list, velocity_list,\
            atom_list, hyp_list, species_list, atom_count


def append_atom_lists(species_list: List[str],
                      position_list: List[np.ndarray],
                      force_list: List[np.ndarray],
                      uncertainty_list: List[np.ndarray],
                      velocity_list: List[np.ndarray],
                      lines: List[str], index: int, noa: int,
                      dft_call: bool, noh: int) -> None:

    """Update lists containing atom information at each snapshot."""

    species, positions, forces, uncertainties, velocities = \
        parse_snapshot(lines, index, noa, dft_call, noh)

    species_list.append(species)
    position_list.append(positions)
    force_list.append(forces)
    uncertainty_list.append(uncertainties)
    velocity_list.append(velocities)


def parse_snapshot(lines, index, noa, dft_call, noh):
    """Parses snapshot of otf output file."""

    # initialize values
    species = []
    positions = np.zeros((noa, 3))
    forces = np.zeros((noa, 3))
    uncertainties = np.zeros((noa, 3))
    velocities = np.zeros((noa, 3))

    # Current setting for # of lines to skip after Frame marker
    skip = 3

    for count, frame_line in enumerate(lines[(index+skip):(index+skip+noa)]):
        # parse frame line
        spec, position, force, uncertainty, velocity = \
            parse_frame_line(frame_line)

        # update values
        species.append(spec)
        positions[count] = position
        forces[count] = force
        uncertainties[count] = uncertainty
        velocities[count] = velocity

    return species, positions, forces, uncertainties, velocities


def parse_frame_line(frame_line):
    """parse a line in otf output.

    :param frame_line: frame line to be parsed
    :type frame_line: string
    :return: species, position, force, uncertainty, and velocity of atom
    :rtype: list, np.arrays
    """

    frame_line = frame_line.split()

    spec = str(frame_line[0])
    position = np.array([float(n) for n in frame_line[1:4]])
    force = np.array([float(n) for n in frame_line[4:7]])
    uncertainty = np.array([float(n) for n in frame_line[7:10]])
    velocity = np.array([float(n) for n in frame_line[10:13]])

    return spec, position, force, uncertainty, velocity
