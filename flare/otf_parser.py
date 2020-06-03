import sys
import numpy as np

from copy import deepcopy
from typing import List, Tuple
from flare import gp, env, struc, otf
from flare.gp import GaussianProcess


class OtfAnalysis:
    def __init__(self, filename, calculate_energy=False):
        self.filename = filename

        self.calculate_energy = calculate_energy

        self.header = parse_header_information(filename)

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

    def make_gp(self, cell=None, call_no=None, hyps=None, init_gp=None,
                hyp_no=None, **kwargs,):

        if call_no is None:
            call_no = len(self.gp_position_list)
        if hyp_no is None:
            hyp_no = call_no
        if hyps is None:
            # check out the last non-empty element from the list
            for icall in reversed(range(hyp_no)):
                if len(self.gp_hyp_list[icall]) > 0:
                    hyps = self.gp_hyp_list[icall][-1]
                    break
        if cell is None:
            cell = self.header['cell']


        if init_gp is None:
            # Use run's values as extracted from header
            # TODO Allow for kernel gradient in header

            dictionary = deepcopy(self.header)
            dictionary['hyps'] = hyps
            for k in kwargs:
                if kwargs[k] is not None:
                    dictionary[k] = kwargs[k]

            gp_model = \
                GaussianProcess.from_dict(dictionary)
        else:
            gp_model = init_gp
            gp_model.hyps = hyps

        for (positions, forces, atoms, _, species) in \
            zip(self.gp_position_list[:call_no],
                self.gp_force_list[:call_no],
                self.gp_atom_list[:call_no], self.gp_hyp_list[:call_no],
                self.gp_species_list[:call_no]):

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
                        atom_strings = line_split[2:-4]
                        for n, atom_string in enumerate(atom_strings):
                            if n == 0:
                                atoms_added.append(int(atom_string[1:-1]))
                            else:
                                atoms_added.append(int(atom_string[0:-1]))

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
                    # catch case where final frame is a DFT call
                    if line_check.startswith('Run complete'):
                        break

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

    def output_md_structures(self):
        """
        Returns structure objects corresponding to the MD frames of an OTF run.
        :return:
        """

        positions = self.position_list
        structures = []
        cell = self.header['cell']
        species = self.header['species']
        forces = self.force_list
        stds = self.uncertainty_list
        for i in range(len(positions)):
            cur_struc = struc.Structure(cell=cell, species=species,
                                        positions=positions[i])
            cur_struc.forces = forces[i]
            cur_struc.stds = stds[i]
            structures.append(cur_struc)
        return structures


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


def strip_and_split(line):
    """
    Helper function which saves a few lines of code elsewhere
    :param line:
    :return:
    """

    line = line.strip().split()
    stripped_line = [subline.strip() for subline in line]

    return stripped_line


def parse_header_information(outfile: str = 'otf_run.out') -> dict:
    """
    Get information about the run from the header of the file
    :param outfile:
    :return:
    """
    with open(outfile, 'r') as f:
        lines = f.readlines()

    header_info = {}

    stopreading = None

    for line in lines:
        if '---' in line or '=' in line:
            stopreading = lines.index(line)
            break

    if stopreading is None:
        raise Exception("OTF File is malformed")

    cutoffs_dict = {}
    for i, line in enumerate(lines[:stopreading]):
        line_lower = line.lower()

        # gp related
        if 'cutoffs' in line_lower:
            line = line.split(':')[1].strip()
            line = line.strip('[').strip(']')
            line = line.split()
            cutoffs = []
            for val in line:
                try:
                    cutoffs.append(float(val))
                except:
                    cutoffs.append(float(val[:-1]))
            header_info['cutoffs'] = cutoffs
        elif 'cutoff' in line_lower:
            line = line.split(':')
            name = line[0][7:]
            value = float(line[1])
            cutoffs_dict[name] = value
            header_info['cutoffs'] = cutoffs_dict

        if 'kernel_name' in line_lower:
            header_info['kernel_name'] = line.split(':')[1].strip()
        elif 'kernels' in line_lower:
            line = line.split(':')[1].strip()
            line = line.strip('[').strip(']')
            line = line.split()
            header_info['kernels'] = line
        elif 'kernel' in line_lower:
            header_info['kernel_name'] = line.split(':')[1].strip()

        if 'number of hyperparameters:' in line_lower:
            header_info['n_hyps'] = int(line.split(':')[1])
        if 'optimization algorithm' in line_lower:
            header_info['algo'] = line.split(':')[1].strip()

        # otf related
        if 'frames' in line_lower:
            header_info['frames'] = int(line.split(':')[1])
        if 'number of atoms' in line_lower:
            header_info['atoms'] = int(line.split(':')[1])
        if 'timestep' in line_lower:
            header_info['dt'] = float(line.split(':')[1])
        if 'system species' in line_lower:
            line = line.split(':')[1]
            line = line.split("'")

            species = [item for item in line if item.isalpha()]

            header_info['species_set'] = set(species)
        if 'periodic cell' in line_lower:
            vectors = []
            for cell_line in lines[i+1:i+4]:
                cell_line = \
                    cell_line.strip().replace('[', '').replace(']', '')
                vec = cell_line.split()
                vector = [float(vec[0]), float(vec[1]), float(vec[2])]
                vectors.append(vector)
            header_info['cell'] = np.array(vectors)
        if 'previous positions' in line_lower:
            struc_spec = []
            prev_positions = []
            for pos_line in lines[i+1:i+1+header_info.get('atoms', 0)]:
                pos = pos_line.split()
                struc_spec.append(pos[0])
                prev_positions.append((float(pos[1]), float(pos[2]),
                                       float(pos[3])))
            header_info['species'] = struc_spec
            header_info['prev_positions'] = np.array(prev_positions)

    return header_info


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
