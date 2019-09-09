import time
import datetime
import numpy as np
import multiprocessing


def write_to_output(string: str, output_file: str = 'otf_run.out'):
    with open(output_file, 'a') as f:
        f.write(string)


def write_header(cutoffs, kernel_name, hyps, algo, dt, Nsteps, structure,
                 output_name, std_tolerance,
                 optional: dict = None):
    with open(output_name, 'w') as f:
        f.write(str(datetime.datetime.now()) + '\n')

    if isinstance(std_tolerance, tuple):
        std_string = "relative uncertainty tolerance: {} eV/A\n".format(
            std_tolerance[0])
        std_string += 'absolute uncertainty tolerance: {} eV/A\n'.format(
            std_tolerance[1])
    elif std_tolerance < 0:
        std_string = \
            'uncertainty tolerance: {} eV/A\n'.format(np.abs(std_tolerance))
    elif std_tolerance > 0:
        std_string = \
            'uncertainty tolerance: {} times noise \n' \
                .format(np.abs(std_tolerance))
    else:
        std_string = ''

    headerstring = ''
    headerstring += \
        'number of cpu cores: {}\n'.format(multiprocessing.cpu_count())
    headerstring += 'cutoffs: {}\n'.format(cutoffs)
    headerstring += 'kernel: {}\n'.format(kernel_name)
    headerstring += 'number of hyperparameters: {}\n'.format(len(hyps))
    headerstring += 'hyperparameters: {}' \
                    '\n'.format(hyps)
    headerstring += 'hyperparameter optimization algorithm: {}' \
                    '\n'.format(algo)
    headerstring += std_string
    headerstring += 'timestep (ps): {}\n'.format(dt)
    headerstring += 'number of frames: {}\n'.format(Nsteps)
    headerstring += 'number of atoms: {}\n'.format(structure.nat)
    headerstring += \
        'system species: {}\n'.format(set(structure.species_labels))
    headerstring += 'periodic cell: \n'
    headerstring += str(structure.cell)

    if optional:
        for key, value in optional.items():
            headerstring += "{}: {} \n".format(key, value)

    # report previous positions
    headerstring += '\nprevious positions (A):\n'
    for i in range(len(structure.positions)):
        headerstring += str(structure.species_labels[i]) + ' '
        for j in range(3):
            headerstring += str("%.8f" % structure.prev_positions[i][j]) + ' '
        headerstring += '\n'
    headerstring += '-' * 80 + '\n'

    write_to_output(headerstring, output_name)


def write_md_config(dt, curr_step, structure, temperature, KE, local_energies,
                    start_time, output_name, dft_step, velocities):
    string = ''

    # Mark if a frame had DFT forces with an asterisk
    if not dft_step:
        string += '-' * 80 + '\n'
        string += "-Frame: " + str(curr_step)
    else:
        string += "\n*-Frame: " + str(curr_step)

    string += '\nSimulation Time: %.3f ps \n' % (dt * curr_step)

    # Construct Header line
    string += 'El  Position (A) \t\t\t\t '
    if not dft_step:
        string += 'GP Force (ev/A) '
    else:
        string += 'DFT Force (ev/A) '
    string += '\t\t\t\t Std. Dev (ev/A) \t'
    string += '\t\t\t\t Velocities (A/ps) \n'

    # Construct atom-by-atom description
    for i in range(len(structure.positions)):
        string += str(structure.species_labels[i]) + ' '
        for j in range(3):
            string += str("%.8f" % structure.positions[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8f" % structure.forces[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % structure.stds[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % velocities[i][j]) + ' '
        string += '\n'

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


def write_hyps(hyp_labels, hyps, start_time, output_name, like, like_grad):
    write_to_output('\nGP hyperparameters: \n', output_name)

    for i, label in enumerate(hyp_labels):
        write_to_output('Hyp{} : {} = {}\n'.format(i, label, hyps[i]),
                        output_name)

    write_to_output('likelihood: ' + str(like) + '\n', output_name)
    write_to_output('likelihood gradient: ' + str(like_grad) + '\n',
                    output_name)
    time_curr = time.time() - start_time
    write_to_output('wall time from start: %.2f s \n' % time_curr,
                    output_name)


def conclude_run(output_name):
    footer = '-' * 20 + '\n'
    footer += 'Run complete. \n'

    write_to_output(footer, output_name)


def write_gp_dft_comparison(curr_step, frame,
                            start_time, dft_forces,
                            mae, mac, local_energies=None, KE=None,
                            output_name=''):
    """
    :param dft_forces:
    :param mae:
    :param mac:
    :param KE:
    :param curr_step:
    :param frame:
    :param local_energies:
    :param start_time:
    :param output_name:
    :return:
    """
    string = ''

    # Mark if a frame had DFT forces with an asterisk
    string += "\n*-Frame: " + str(curr_step)

    # Construct Header line
    string += '\nEl  Position (A) \t\t\t\t '
    string += 'GP Force (ev/A)  \t\t\t\t'
    string += 'Std. Dev (ev/A) \t\t\t\t'
    string += 'DFT Force (ev/A)  \t\t\t\t \n'

    # Construct atom-by-atom description
    for i in range(len(frame.positions)):
        string += str(frame.species_labels[i]) + ' '
        for j in range(3):
            string += str("%.8f" % frame.positions[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8f" % frame.forces[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % frame.stds[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8f" % dft_forces[i][j]) + ' '
        string += '\n'

    string += '\n'

    string += 'mean absolute error: %.2f meV/A \n' % mae
    string += 'mean absolute dft component: %.2f meV/A \n' % mac

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
