import sys
import numpy as np
from typing import List, Tuple
from flare import gp, env, struc, kernels, otf


class OtfAnalysis:
    def __init__(self, filename, calculate_energy=False):
        self.filename = filename
        self.calculate_energy = calculate_energy
        self.header = parse_header_information(filename)

        position_list, species_list, gp_force_list, dft_force_list, \
            gp_uncertainty_list, update_frames, msds, energies, \
            atom_list, hyp_list = \
            self.parse_file(filename)

        self.position_list = position_list
        self.species_list = species_list
        self.gp_force_list = gp_force_list
        self.dft_force_list = dft_force_list
        self.gp_uncertainty_list = gp_uncertainty_list
        self.update_frames = update_frames
        self.msds = msds
        self.atom_list = atom_list
        self.hyp_list = hyp_list

        # assumption: species list is the same across AIMD simulation
        self.species_labels = species_list[0]
        _, coded_species = struc.get_unique_species(self.species_labels)
        self.species = coded_species

        if self.calculate_energy:
            self.energies = energies

    def make_gp(self, cell=None, kernel=None, kernel_grad=None, algo=None,
                call_no=None, cutoffs=None, hyps=None, init_gp=None,
                energy_force_kernel=None):

        if init_gp is None:
            # Use run's values as extracted from header
            # TODO Allow for kernel gradient in header
            if cell is None:
                cell = self.header['cell']
            if kernel is None:
                kernel = self.header['kernel']
            if kernel_grad is None:
                raise Exception('Kernel gradient not supplied')
            if algo is None:
                algo = self.header['algo']
            if cutoffs is None:
                cutoffs = self.header['cutoffs']
            if call_no is None:
                call_no = len(self.update_frames)
            if hyps is None:
                gp_hyps = self.hyp_list[call_no-1]
            else:
                gp_hyps = hyps

            gp_model = \
                gp.GaussianProcess(kernel, kernel_grad, gp_hyps,
                                   cutoffs, opt_algorithm=algo,
                                   energy_force_kernel=energy_force_kernel)
        else:
            gp_model = init_gp
            call_no = len(self.update_frames)
            gp_hyps = self.hyp_list[call_no-1][-1]
            gp_model.hyps = gp_hyps

        for n, frame in enumerate(self.update_frames):
            positions = self.position_list[frame]
            struc_curr = \
                struc.Structure(cell, self.species, positions,
                                species_labels=self.species_labels)
            atoms = self.atom_list[n]
            forces = self.dft_force_list[frame]
            gp_model.update_db(struc_curr, forces, custom_range=atoms)

        gp_model.set_L_alpha()

        return gp_model

    def parse_file(self, filename):
        """
        Exclusively parses MD run information
        :param filename:
        :return:
        """
        position_list = []
        species_list = []
        gp_force_list = []
        dft_force_list = []
        gp_uncertainty_list = []
        update_frames = []
        msds = []
        energies = []
        atom_list = []
        hyp_list = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        for index, line in enumerate(lines):
            if line.startswith("number of atoms"):
                at_line = line.split()
                noa = int(at_line[3])

            # number of hyperparameters
            if line.startswith("number of hyperparameters"):
                line_curr = line.split(':')
                noh = int(line_curr[-1])

            # DFT forces
            if line.startswith("*-Frame"):
                species, positions, forces, _ = \
                    parse_snapshot(lines, index, noa)

                position_list.append(positions)
                species_list.append(species)
                msds.append(np.mean((positions - position_list[0])**2))
                dft_force_list.append(forces)

                frame_line = line.split()
                frame_curr = (int(frame_line[1]))

            # GP forces
            if line.startswith("-Frame"):
                _, _, forces, uncertainties = \
                    parse_snapshot(lines, index, noa)

                gp_force_list.append(forces)
                gp_uncertainty_list.append(uncertainties)

                if self.calculate_energy:
                    en_line = lines[index+5+noa].split()
                    energies.append(float(en_line[2]))

            # Update frame
            if line.startswith("Adding atom"):
                # keep track of atoms added to training set
                atoms_added = []
                line_split = line.split()
                atom_strings = line_split[2:-4]
                for n, atom_string in enumerate(atom_strings):
                    if n == 0:
                        atoms_added.append(int(atom_string[1:-1]))
                    else:
                        atoms_added.append(int(atom_string[0:-1]))
                atom_list.append(atoms_added)
                update_frames.append(frame_curr)

            # Hyperparameter block
            if line.startswith("GP hyperparameters:"):
                hyps = []
                for hyp_line in lines[(index+1):(index+1+noh)]:
                    hyp_line = hyp_line.split()
                    hyps.append(float(hyp_line[-1]))
                hyps = np.array(hyps)
                hyp_list.append(hyps)

        return position_list, species_list, gp_force_list, dft_force_list, \
            gp_uncertainty_list, update_frames, msds, energies, \
            atom_list, hyp_list


def parse_snapshot(lines, index, noa):
    """Parses snapshot of otf output file."""

    # initialize values
    species = []
    positions = np.zeros((noa, 3))
    forces = np.zeros((noa, 3))
    uncertainties = np.zeros((noa, 3))

    # Current setting for # of lines to skip after Frame marker
    skip = 3

    for count, frame_line in enumerate(lines[(index+skip):(index+skip+noa)]):
        # parse frame line
        spec, position, force, uncertainty = \
            parse_frame_line(frame_line)

        # update values
        species.append(spec)
        positions[count] = position
        forces[count] = force
        uncertainties[count] = uncertainty

    return species, positions, forces, uncertainties


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

    for i, line in enumerate(lines[:stopreading]):
        # TODO Update this in full
        if 'cutoffs' in line:
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
        if 'frames' in line:
            header_info['frames'] = int(line.split(':')[1])
        if 'kernel' in line:
            header_info['kernel'] = line.split(':')[1].strip()
        if 'number of hyperparameters:' in line:
            header_info['n_hyps'] = int(line.split(':')[1])
        if 'optimization algorithm' in line:
            header_info['algo'] = line.split(':')[1].strip()
        if 'number of atoms' in line:
            header_info['atoms'] = int(line.split(':')[1])
        if 'timestep' in line:
            header_info['dt'] = float(line.split(':')[1])
        if 'system species' in line:
            line = line.split(':')[1]
            line = line.split("'")

            species = [item for item in line if item.isalpha()]

            header_info['species_set'] = set(species)
        if 'periodic cell' in line:
            vectors = []
            for cell_line in lines[i+1:i+4]:
                cell_line = \
                    cell_line.strip().replace('[', '').replace(']', '')
                vec = cell_line.split()
                vector = [float(vec[0]), float(vec[1]), float(vec[2])]
                vectors.append(vector)
            header_info['cell'] = np.array(vectors)
        if 'previous positions' in line:
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

    return spec, position, force, uncertainty
