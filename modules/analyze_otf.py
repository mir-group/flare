#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from typing import List


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

    for line in lines[:stopreading]:

        if 'Frames' in line:
            header_info['frames'] = int(line.split(':')[1])
        if 'Kernel' in line:
            header_info['kernel'] = line.split(':')[1].strip()
        if 'Hyperparameters' in line:
            header_info['n_hyps'] = int(line.split(':')[1])
        if 'Optimization Algorithm' in line:
            header_info['algo'] = line.split(':')[1].strip()
        if 'Atoms' in line:
            header_info['atoms'] = int(line.split(':')[1])
        if 'Cutoff' in line:
            header_info['cutoff'] = float(line.split(':')[1])
        if 'Timestep' in line:
            header_info['dt'] = float(line.split(':')[1])
        if 'Species' in line:
            line = line.split(':')[1]
            line = line.split("'")

            species = [item for item in line if item.isalpha()]

            header_info['species'] = set(species)

    return header_info


def parse_md_information(outfile: str = 'otf_run.out') -> (List[str], np.array,
                                                           np.array, np.array):
    """
    Parse information about the OTF MD run
    :param outfile:
    :return:
    """
    with open(outfile, 'r') as f:
        lines = f.readlines()

    frame_indices = [lines.index(line) + 2 for line in lines if line[0] == '-']

    # Parse number of atoms and number of timesteps
    md_nat_line = [line.split(':')[-1] for line in lines[0:20] if
                   'Number of Atoms' in line]
    md_nat = int(md_nat_line[0])

    md_steps_line = [line.split(':')[-1] for line in lines[0:20] if
                     'Number of Frames' in line]
    md_steps = int(md_steps_line[0])

    # Number of atoms is constant between frames and number of frames is known
    run_positions = np.empty(shape=(md_steps, md_nat, 3))
    run_forces = np.empty(shape=(md_steps, md_nat, 3))
    run_stds = np.empty(shape=(md_steps, md_nat, 3))

    # Assume species order is constant for MD run
    # Get species from first frame
    species = []
    first_frame = frame_indices[0]
    for n in range(md_nat):
        specie = strip_and_split(lines[first_frame + n])[0]
        species.append(specie)

    # Get frame information and populate arrays
    for frame_n, frame_index in enumerate(frame_indices):

        for at in range(md_nat):
            data = strip_and_split(lines[frame_index + at])
            run_positions[frame_n, at, :] = data[1:4]
            run_forces[frame_n, at, :] = data[4:7]
            run_stds[frame_n, at, :] = data[7:10]

    return species, run_positions, run_forces, run_stds


def parse_dft_information(outfile: str) -> (List[str], List[np.array], \
                                            List[np.array]):
    """
    Parse the output of a otf run for analysis
    :param outfile: str, Path to file
    :return: dict{int:value,'species':list}, Dict of positions, forces,
    vars indexed by frame and of species
    """

    with open(outfile, 'r') as f:
        lines = f.readlines()

    # DFT indices start with ~ and are 2 lines above the data
    dft_indices = [lines.index(line) + 3 for line in lines if line[0] == '~']
    # Frame indices start with - and are 2 lines above the data

    # Species and positions may vary from DFT run to DFT run

    # TODO turn into numpy array and extend array with each new data point;
    # would be more efficient for large files

    dft_lattices = []
    dft_species = []
    dft_positions = []
    dft_forces = []

    for frame_n, frame_index in enumerate(dft_indices):

        dft_run_nat = 0

        dft_run_lattice = []
        dft_run_species = []
        dft_run_positions = []
        dft_run_forces = []

        # Read lattice in
        lattice_line = lines[frame_index - 2].strip().split(':')[1]
        lattice_line = lattice_line.strip().strip('[').strip(']')
        lattice_sublines = lattice_line.split(',')

        for vec in lattice_sublines:
            vec = vec.strip().strip('[').strip(']')
            dft_run_lattice.append(np.fromstring(vec, dtype=float, sep=' '))

        dft_lattices.append(np.array(dft_run_lattice))

        # Read atoms, positions, forces in

        while lines[frame_index + dft_run_nat][0] != '=':
            dft_run_nat += 1

        for at in range(dft_run_nat):
            data = strip_and_split(lines[frame_index + at])

            dft_run_species.append(data[0].strip(':'))

            curr_position = np.array([data[1], data[2], data[3]],
                                     dtype='float')
            curr_force = np.array([data[4], data[5], data[6]], dtype='float')

            dft_run_positions.append(curr_position)
            dft_run_forces.append(curr_force)

        dft_species.append(dft_run_species)
        dft_positions.append(np.array(dft_run_positions))
        dft_forces.append(np.array(dft_run_forces))

    return dft_lattices, dft_species, dft_positions, dft_forces


def parse_hyperparameter_information(outfile: str = 'otf_run.out')-> List[
    np.array]:

    with open(outfile, 'r') as f:
        lines = f.readlines()

    header_info = parse_header_information(outfile)
    n_hyps = header_info['n_hyps']

    hyp_indices = [i+1 for i,text in enumerate(lines) if 'New GP Hyper' in
                   text]

    hyperparameter_set = [np.zeros(n_hyps)]*len(hyp_indices)

    for set_index,line_index in enumerate(hyp_indices):

        for j in range(n_hyps):

            cur_hyp = lines[line_index+j].strip().split('=')[1]

            hyperparameter_set[set_index][j] = float(cur_hyp)

    return hyperparameter_set


def parse_time_information(outfile: str = 'otf_run.out'):
    # with open(outfile, 'r') as f:
    #    lines = f.readlines()

    raise NotImplementedError


if __name__ == '__main__':
    pass
