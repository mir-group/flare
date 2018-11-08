import numpy as np
import sys
import os
import qe_input
import qe_parsers
import datetime
import time


def convergence(txt_name, input_file_string, output_file_string, pw_loc,
                calculation, scf_inputs, nks, ecutwfcs, rho_facs,
                electron_maxstep=100):
    # initialize update text
    txt = update_init()
    write_file(txt_name, txt)

    # converged run
    initial_input_name = input_file_string + '.in'
    initial_output_name = output_file_string + '.out'

    scf = qe_input.QEInput(initial_input_name, initial_output_name, pw_loc,
                           calculation, scf_inputs,
                           electron_maxstep=electron_maxstep)

    # perform and time converged scf run
    time0 = time.time()
    scf.run_espresso(npool=False)
    time1 = time.time()
    scf_time = time1 - time0

    conv_energy, conv_forces = qe_parsers.parse_scf(initial_output_name)

    txt += record_results(scf_inputs['kvec'][0], scf_inputs['ecutwfc'],
                          scf_inputs['ecutrho'], conv_energy, conv_forces,
                          conv_energy, conv_forces, scf_time)
    write_file(txt_name, txt)

    # loop over parameters
    for nk in nks:
        for ecutwfc in ecutwfcs:
            for rho_fac in rho_facs:
                ecutrho = ecutwfc * rho_fac
                scf_inputs['kvec'] = np.array([nk, nk, nk])
                scf_inputs['ecutrho'] = ecutrho
                scf_inputs['ecutwfc'] = ecutwfc

                input_name = input_file_string + \
                    '_nk%i_e%i_rho%i.in' % (nk, ecutwfc, rho_fac)
                output_name = output_file_string + \
                    '_nk%i_e%i_rho%i.out' % (nk, ecutwfc, rho_fac)

                scf = qe_input.QEInput(input_name, output_name,
                                       pw_loc, calculation, scf_inputs,
                                       electron_maxstep=electron_maxstep)

                # perform and time scf run
                time0 = time.time()
                scf.run_espresso(npool=False)
                time1 = time.time()
                scf_time = time1 - time0

                energy, forces = qe_parsers.parse_scf(output_name)

                txt += record_results(nk, ecutwfc, ecutrho, energy, forces,
                                      conv_energy, conv_forces, scf_time)
                write_file(txt_name, txt)

    txt += update_fin()
    write_file(txt_name, txt)

    # remove output directory
    if os.path.isdir('output'):
        os.system('rm -r output')
    if os.path.isdir('__pycache__'):
        os.system('rm -r __pycache__')


def reshape_forces(forces: list) -> np.ndarray:
    forces_array = np.array(forces)
    forces_array = forces_array.reshape(forces_array.size)
    return forces_array


def print_results(nk, ecutwfc, ecutrho, energy, forces, conv_energy,
                  conv_forces):
    print('\n')
    print('nk: %i' % nk)
    print('ecutwfc: %.2f' % ecutwfc)
    print('ecutrho: %.2f' % ecutrho)
    print('energy: %f' % energy)

    en_diff = energy - conv_energy
    print('energy difference from converged value: %.2e eV' % en_diff)

    # reshape converged force
    conv_forces_array = reshape_forces(conv_forces)
    forces_array = reshape_forces(forces)
    force_diff = conv_forces_array - forces_array
    force_MAE = np.mean(np.abs(force_diff))
    print('force MAE: %.2e eV/A' % force_MAE)

    max_err = np.max(np.abs(force_diff))
    print('max force error: %.2e eV/A' % max_err)
    print('\n')


def record_results(nk, ecutwfc, ecutrho, energy, forces, conv_energy,
                   conv_forces, scf_time):
    txt = """

Inputs:
nk: %i
ecutwfc: %.2f
ecutrho: %.2f

Convergence results:
energy: %f eV.""" % (nk, ecutwfc, ecutrho, energy)

    en_diff = energy - conv_energy
    txt += """
energy difference from converged value: %.2e eV.""" % en_diff

    # reshape converged force
    conv_forces_array = reshape_forces(conv_forces)
    forces_array = reshape_forces(forces)
    force_diff = conv_forces_array - forces_array
    force_MAE = np.mean(np.abs(force_diff))
    txt += """
force MAE: %.2e eV/A.""" % force_MAE

    max_err = np.max(np.abs(force_diff))
    txt += """
max force error: %.2e eV/A.
scf time: %.2f s.
""" % (max_err, scf_time)

    return txt


# -----------------------------------------------------------------------------
#                          monitor code progress
# -----------------------------------------------------------------------------

def write_file(fname, text):
    with open(fname, 'w') as fin:
        fin.write(text)


def update_init():
    init_text = """Quantum Espresso convergence test: %s.
Author: Jonathan Vandermause.
""" % str(datetime.datetime.now())

    return init_text


def update_fin():
    fin_text = """
-------------------------------------------------------------------------------
   JOB DONE.
-------------------------------------------------------------------------------
"""
    return fin_text

if __name__ == '__main__':
    nk = 5
    ecutwfc = 50
    ecutrho = 100
    energy = 24.5
    forces = [np.array([1, 2, 3])]
    conv_energy = 25
    conv_forces = [np.array([4, 5, 6])]
    record_test = record_results(nk, ecutwfc, ecutrho, energy, forces,
                                 conv_energy, conv_forces, 4)
    print(record_test)
