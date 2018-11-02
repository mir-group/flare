import numpy as np
import sys
import os
import qe_input
import qe_parsers


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
    print('forces:')
    print(forces)

    en_diff = energy - conv_energy
    print('energy difference from converged value: %.2e eV' % en_diff)

    # reshape converged force
    conv_forces_array = reshape_forces(conv_forces)
    forces_array = reshape_forces(forces)
    force_diff = conv_forces_array - forces_array
    force_MAE = np.mean(np.abs(force_diff))
    print('force MAE: %.2e eV/A' % force_MAE)
    print('\n')


def convergence(input_file_name, output_file_name, pw_loc,
                calculation, scf_inputs, nks, ecutwfcs, rho_facs):

    # converged run
    scf = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                           calculation, scf_inputs)
    scf.run_espresso(npool=False)
    conv_energy, conv_forces = qe_parsers.parse_scf(output_file_name)

    print('converged values:')
    print_results(scf_inputs['kvec'][0], scf_inputs['ecutwfc'],
                  scf_inputs['ecutrho'], conv_energy, conv_forces,
                  conv_energy, conv_forces)

    # loop over parameters
    for nk in nks:
        for ecutwfc in ecutwfcs:
            for rho_fac in rho_facs:
                ecutrho = ecutwfc * rho_fac
                scf_inputs['kvec'] = np.array([nk, nk, nk])
                scf_inputs['ecutrho'] = ecutrho
                scf_inputs['ecutwfc'] = ecutwfc

                scf = qe_input.QEInput(input_file_name, output_file_name,
                                       pw_loc, calculation, scf_inputs)
                scf.run_espresso(npool=False)
                energy, forces = qe_parsers.parse_scf(output_file_name)
                print_results(nk, ecutwfc, ecutrho, energy, forces,
                              conv_energy, conv_forces)

    # remove output directory
    if os.path.isdir('output'):
        os.system('rm -r output')
    if os.path.isdir('__pycache__'):
        os.system('rm -r __pycache__')
