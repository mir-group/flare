import numpy as np
import sys
import os
import qe_input
import qe_parsers


def print_results(nk, ecutwfc, ecutrho, energy, conv_energy):
    print('\n')
    print('nk: '+str(nk))
    print('ecutwfc: '+str(ecutwfc))
    print('ecutrho: '+str(ecutrho))
    print('energy: '+str(energy))
    print('difference from converged value: '+str(energy-conv_energy))
    print('\n')


def convergence(input_file_name, output_file_name, pw_loc,
                calculation, scf_inputs, nks, ecutwfcs, rho_facs):

    # converged run
    scf = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                           calculation, scf_inputs)
    scf.run_espresso()
    conv_energy = qe_parsers.parse_scf_energy(output_file_name)

    print('converged values:')
    print_results(scf_inputs['kvec'][0], scf_inputs['ecutwfc'],
                  scf_inputs['ecutrho'], conv_energy, conv_energy)

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
                scf.run_espresso()
                energy = qe_parsers.parse_scf_energy(output_file_name)
                print_results(nk, ecutwfc, ecutrho, energy, conv_energy)

    # remove output directory
    if os.path.isdir('output'):
        os.system('rm -r output')
    if os.path.isdir('__pycache__'):
        os.system('rm -r __pycache__')
