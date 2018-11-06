#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
OTF Parsing and Analysis Suite

Steven Torrisi
"""

import numpy as np
from typing import List


def strip_and_split(line):
    """
    Helper function which saves a few lines of code elsewhere
    :param line:
    :return:
    """

    line = line.strip()
    line = line.split()
    stripped_line = [subline.strip() for subline in line]

    return stripped_line


def parse_md_information(outfile: str = 'otf_run.out') -> (List[str], np.array,
                                                           np.array, np.array):
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


def parse_dft_information(outfile: str):
    """
    Parse the output of a otf run for analysis
    :param outfile: str, Path to file
    :return: dict{int:value,'species':list}, Dict of positions, forces,
    vars indexed by frame and of species
    """

    with open(outfile, 'r') as f:
        lines = f.readlines()

    # DFT indices start with ~ and are 2 lines above the data
    dft_indices = [lines.index(line) + 2 for line in lines if line[0] == '~']
    # Frame indices start with - and are 2 lines above the data

    # Species and positions may vary from DFT run to DFT run
    # TODO turn into numpy array and extend array with each new data point
    dft_species = []
    dft_positions = []
    dft_forces = []

    for frame_n, frame_index in enumerate(dft_indices):

        dft_run_nat = 0

        dft_run_species = []
        dft_run_positions = []
        dft_run_forces = []

        while lines[frame_index + dft_run_nat][0] != '=':
            dft_run_nat += 1

        for at in range(dft_run_nat):
            data = strip_and_split(lines[frame_index])

            dft_run_species.append(data[0])

            curr_position = np.array([data[1], data[2], data[3]])
            curr_force = np.array([data[4], data[5], data[6]])

            dft_run_positions.append(curr_position)
            dft_run_forces.append(curr_force)

        dft_species.append(dft_run_species)
        dft_positions.append(np.array(dft_run_positions))
        dft_forces.append(np.array(dft_run_forces))

    return dft_species, dft_positions, dft_forces


# TODO Implement these
def parse_hyperparameter_information(outfile: str = 'otf_run.out'):
    with open(outfile, 'r') as f:
        lines = f.readlines()

    raise NotImplementedError


def parse_time_information(outfile: str = 'otf_run.out'):
    with open(outfile, 'r') as f:
        lines = f.readlines()

    raise NotImplementedError


if __name__ == '__main__':
    pass
