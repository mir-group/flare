import sys
import numpy as np
import crystals
from typing import List, Tuple
sys.path.append('../otf_engine')
import gp, env, struc, kernels, otf


class OtfAnalysis:
    def __init__(self, filename):
        self.filename = filename

        position_list, force_list, uncertainty_list, velocity_list,\
            dft_frames, temperatures, times, msds, dft_times = \
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

        gp_position_list, gp_force_list, gp_uncertainty_list,\
            gp_velocity_list, gp_atom_list, gp_hyp_list, gp_cutoff_radius,\
            gp_species_list = self.extract_gp_info(filename)

        self.gp_position_list = gp_position_list
        self.gp_force_list = gp_force_list
        self.gp_uncertainty_list = gp_uncertainty_list
        self.gp_velocity_list = gp_velocity_list
        self.gp_atom_list = gp_atom_list
        self.gp_hyp_list = gp_hyp_list
        self.gp_cutoff_radius = gp_cutoff_radius
        self.gp_species_list = gp_species_list

    def make_gp(self, cell, kernel, bodies, algo, call_no, cutoffs=None):
        gp_model = gp.GaussianProcess(kernel, bodies, algo, cutoffs=cutoffs)
        gp_hyps = self.gp_hyp_list[call_no-1]
        gp_model.hyps = gp_hyps

        for count, (positions, forces, atom, _, species) in \
            enumerate(zip(self.gp_position_list, self.gp_force_list,
                          self.gp_atom_list, self.gp_hyp_list,
                          self.gp_species_list)):
            if count < call_no:
                struc_curr = struc.Structure(cell, species, positions,
                                             self.gp_cutoff_radius)

                gp_model.update_db(struc_curr, forces, custom_range=[atom])

        gp_model.set_L_alpha()

        return gp_model

    @staticmethod
    def get_gp_activation(gp_model):
        pass

    def parse_pos_otf(self, filename):
        position_list = []
        force_list = []
        uncertainty_list = []
        velocity_list = []
        temperatures = []
        dft_frames = []
        dft_times = []
        times = []
        msds = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        n_steps = 0

        for index, line in enumerate(lines):
            if line.startswith("Number of Atoms"):
                at_line = line.split()
                noa = int(at_line[3])

            # number of hyperparameters
            if line.startswith("# Hyperparameters"):
                line_curr = line.split()
                noh = int(line_curr[2])

            if line.startswith("-*"):
                dft_line = line.split()
                dft_frames.append(int(dft_line[1]))
                dft_times.append(float(dft_line[4]))

            if line.startswith("- Frame"):
                n_steps += 1
                frame_line = line.split()
                sim_time = float(frame_line[5])
                times.append(sim_time)

                _, positions, forces, uncertainties, velocities = \
                    parse_snapshot(lines, index, noa, False, noh)

                position_list.append(positions)
                force_list.append(forces)
                uncertainty_list.append(uncertainties)
                velocity_list.append(velocities)

                temp_line = lines[index+2+noa].split()
                temperatures.append(float(temp_line[1]))

                msds.append(np.mean((positions - position_list[0])**2))

        return position_list, force_list, uncertainty_list, velocity_list,\
            dft_frames, temperatures, times, msds, dft_times

    def extract_gp_info(self, filename):
        species_list = []
        position_list = []
        force_list = []
        uncertainty_list = []
        velocity_list = []
        atom_list = []
        hyp_list = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        for index, line in enumerate(lines):
            # cutoff radius
            if line.startswith("Structure Cutoff Radius:"):
                line_curr = line.split()
                cutoff_radius = float(line_curr[3])

            # number of atoms
            if line.startswith("Number of Atoms"):
                line_curr = line.split()
                noa = int(line_curr[3])

            # number of hyperparameters
            if line.startswith("# Hyperparameters"):
                line_curr = line.split()
                noh = int(line_curr[2])

            # TODO: change otf program so that this if statement is unnecessary
            if line.startswith("Calling DFT with training atoms"):
                line_curr = line.split()

                # TODO: write function for updating hyps list
                hyps = []
                for frame_line in lines[(index+4):(index+4+noh)]:
                    frame_line = frame_line.split()
                    hyps.append(float(frame_line[5]))
                hyps = np.array(hyps)
                hyp_list.append(hyps)
                hyp_list.append(hyps)

                # TODO: generalize this to account for arbitrary starting list
                append_atom_lists(species_list, position_list, force_list,
                                  uncertainty_list, velocity_list,
                                  lines, index, noa, True, noh)
                append_atom_lists(species_list, position_list, force_list,
                                  uncertainty_list, velocity_list,
                                  lines, index, noa, True, noh)

                atom_list.append(0)
                atom_list.append(30)

            if line.startswith("Calling DFT due to"):
                line_curr = line.split()

                # TODO: write function for updating hyps list
                hyps = []
                for frame_line in lines[(index+4):(index+4+noh)]:
                    frame_line = frame_line.split()
                    hyps.append(float(frame_line[5]))
                hyps = np.array(hyps)
                hyp_list.append(hyps)

                # TODO: generalize atom list update to account for arbitrary
                # list
                append_atom_lists(species_list, position_list, force_list,
                                  uncertainty_list, velocity_list,
                                  lines, index, noa, True, noh)
                atom_list.append(int(line_curr[5]))

        return position_list, force_list, uncertainty_list, velocity_list,\
            atom_list, hyp_list, cutoff_radius,\
            species_list


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

    # number of lines to skip after frame line
    if dft_call is True:
        skip = 9 + noh
    else:
        skip = 2

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
