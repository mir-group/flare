import time
import datetime
import numpy as np


def write_to_output(string: str, output_file: str = 'otf_run.out'):
    with open(output_file, 'a') as f:
        f.write(string)


def write_header(cutoffs, kernel_name, hyps, algo, dt, Nsteps, structure,
                 output_name):

    with open(output_name, 'w') as f:
        f.write(str(datetime.datetime.now()) + '\n')

    headerstring = ''
    headerstring += 'Cutoffs: {}\n'.format(cutoffs)
    headerstring += 'Kernel: {}\n'.format(kernel_name)
    headerstring += '# Hyperparameters: {}\n'.format(len(hyps))
    headerstring += 'Hyperparameters: {}' \
                    '\n'.format(hyps)
    headerstring += 'Hyperparameter Optimization Algorithm: {}' \
                    '\n'.format(algo)
    headerstring += 'Timestep (ps): {}\n'.format(dt)
    headerstring += 'Number of Frames: {}\n'.format(Nsteps)
    headerstring += 'Number of Atoms: {}\n'.format(structure.nat)
    headerstring += 'System Species: {}\n'.format(set(structure.species))
    headerstring += 'Periodic cell: \n'
    headerstring += str(structure.cell)
    headerstring += '\n'

    # report previous positions
    headerstring += 'Previous Positions (A): \n'
    for i in range(len(structure.positions)):
        headerstring += structure.species[i] + ' '
        for j in range(3):
            headerstring += str("%.8f" % structure.prev_positions[i][j]) + ' '
        headerstring += '\n'

    write_to_output(headerstring, output_name)


def write_md_config(dt, curr_step, structure, temperature, KE, local_energies,
                    start_time, output_name):
    string = "-------------------- \n"

    # Mark if a frame had DFT forces with an asterisk
    if not structure.dft_forces:
        string += "-Frame: " + str(curr_step)
    else:
        string += "*-Frame: " + str(curr_step)

    string += ' Simulation Time: %.3f ps \n' % (dt * curr_step)

    # Construct Header line
    string += 'El \t\t\t  Position (A) \t\t\t\t\t '
    if not structure.dft_forces:
        string += 'GP Force (ev/A) '
    else:
        string += 'DFT Force (ev/A) '
    string += '\t\t\t\t\t\t Std. Dev (ev/A) \n'

    # Construct atom-by-atom description
    for i in range(len(structure.positions)):
        string += structure.species[i] + ' '
        for j in range(3):
            string += str("%.8f" % structure.positions[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8f" % structure.forces[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % structure.stds[i][j]) + ' '
        string += '\n'

    string += 'temperature: %.2f K \n' % temperature
    string += 'kinetic energy: %.6f eV \n' % KE

    # calculate potential and total energy
    if local_energies is not None:
        pot_en = np.sum(local_energies)
        tot_en = KE + pot_en
        string += \
            'potential energy: %.6f eV \n' % pot_en
        string += 'total energy: %.6f eV \n' % tot_en

    string += 'wall time from start: %.2f s \n' % \
        (time.time() - start_time)

    write_to_output(string, output_name)


def write_hyps(hyp_labels, hyps, start_time, dft_count, output_name):
    write_to_output('New GP Hyperparameters: \n', output_name)

    for i, label in enumerate(hyp_labels):
        write_to_output('Hyp{} : {} = {}\n'.format(i, label, hyps[i]),
                        output_name)
    time_curr = time.time() - start_time
    write_to_output('wall time from start: %.2f s \n' % time_curr,
                    output_name)
    write_to_output('number of DFT calls: %i \n' % dft_count,
                    output_name)


def conclude_run(output_name):
    footer = '▬' * 20 + '\n'
    footer += 'Run complete. \n'

    write_to_output(footer, output_name)
