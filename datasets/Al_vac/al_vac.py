#!/usr/bin/env python3
# pylint: disable=redefined-outer-name

"""" OTF Test Case - Bulk Al with a vacancy

Simon Batzner
"""
import os

import numpy as np

from ase.build import *
from ase.spacegroup import crystal

from qe_parsers import parse_md_output
from otf import OTF


def create_structure(el, alat, size):
    # create bulk cell
    unit_cell = crystal(el, [(0, 0, 0)], spacegroup=225, cellpar=[alat, alat, alat, 90, 90, 90])

    # make supercell
    multiplier = np.identity(3) * size
    supercell = make_supercell(unit_cell, multiplier)

    # remove atom
    supercell.pop(supercell.get_number_of_atoms() // 2)
    cell = supercell.get_cell()
    al_pos = np.asarray(supercell.positions)
    nat = supercell.get_number_of_atoms()

    return cell, al_pos, nat


def run_test_md():
    input_file_name = './Al_scf.in'
    output_file_name = './Al_scf.out'
    pw_loc = os.environ['PWSCF_COMMAND']

    qe_command = '{0} < {1} > {2}'.format(pw_loc, input_file_name, output_file_name)
    os.system(qe_command)

    return parse_md_output(output_file_name)


def run_otf_md(input_file):
    output_file_name = './Al_OTF.out'

    # setup run from QE input file
    Al_OTF = OTF(qe_input=input_file,
                     dt=0.0001,
                     number_of_steps=1000,
                     kernel='two_body',
                     cutoff=3)

    # run otf
    Al_OTF.run()

    return parse_md_output(output_file_name)

if __name__ == '__main__':
    workdir = os.getcwd()
    print(os.environ['ESPRESSO_PSEUDO'])
    print(os.environ['PWSCF_COMMAND'])
    print(workdir)

    # # params
    # el = 'Al'
    # size = 2
    # ecut = 38
    # alat = 3.978153546
    # nk = 7
    #
    # # debug mode
    # local = True
    # ecut = 5
    # nk = 1

    # run otf
    input_file_name = './Al_scf.in'
    results = run_otf_md(input_file=input_file_name)



