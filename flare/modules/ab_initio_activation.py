import numpy as np
import copy
import sys
import os
from subprocess import call
import datetime

# modules
import crystals
import qe_parsers

# otf files
import struc
import qe_util


def vac_diff_fcc_dft(qe_input, fcc_cell, no_runs, species, pw_loc):

    # create text file
    with open('barrier.txt', 'w') as f:
        f.write('barrier calculation \n')
        f.write(str(datetime.datetime.now()) + '\n')

    # create 2x2x2 fcc supercell with atom 0 removed
    alat = fcc_cell[0, 0]
    fcc_unit = crystals.fcc_positions(alat)
    fcc_super = crystals.get_supercell_positions(2, fcc_cell, fcc_unit)
    vac_super = copy.deepcopy(fcc_super)
    vac_super.pop(0)
    vac_super = np.array(vac_super)

    # create list of positions for the migrating atom
    start_pos = vac_super[0]
    end_pos = np.array([0, 0, 0])
    diff_vec = end_pos - start_pos
    test_list = []
    step = diff_vec / (no_runs - 1)
    for n in range(no_runs):
        test_list.append(start_pos + n*step)

    # for each position, create a structure and run dft
    for test_pos in test_list:
        vac_super[0] = test_pos
        struc_curr = struc.Structure(fcc_cell*2, species, vac_super,
                                     cutoff=None)

        xdist_curr = test_pos[0]
        output_file_name = 'al_vac_%.3f.out' % xdist_curr
        forces_curr, en_curr = run_espresso(qe_input, struc_curr, pw_loc,
                                            output_file_name)

        # record results in output file
        with open('barrier.txt', 'a') as f:
            f.write('position: \n')
            f.write(str(vac_super))
            f.write('\n')
            f.write('forces (eV/A): \n')
            f.write(str(forces_curr))
            f.write('\n')
            f.write('energy (Ry): \n')
            f.write(str(en_curr))
            f.write('\n')


def run_espresso(qe_input, structure, pw_loc, output_file_name):
    run_qe_path = qe_input
    # os.system(' '.join(['cp', qe_input, run_qe_path]))
    call(' '.join(['cp', qe_input, run_qe_path]), shell=True)
    qe_util.edit_qe_input_positions(run_qe_path, structure)
    qe_command = 'mpirun {0} < {1} > {2}'.format(pw_loc, run_qe_path,
                                                 output_file_name)
    # os.system(qe_command)
    call(qe_command, shell=True)

    forces, total_energy = qe_util.parse_qe_forces_and_energy(output_file_name)

    return forces, total_energy
