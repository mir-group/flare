import numpy as np


def parse_barrier_file(barrier_file):
    with open(barrier_file, 'r') as outf:
        lines = outf.readlines()

    energies = []
    forces = []
    positions = []

    energy_lines = []
    force_lines = []
    pos_lines = []

    for count, line in enumerate(lines):
        if line.startswith('energy'):
            energy_lines.append(count)

        if line.startswith('forces'):
            force_lines.append(count)

        if line.startswith('position'):
            pos_lines.append(count)

    noa = energy_lines[0] - force_lines[0] - 1

    for energy_line, force_line, pos_line in zip(energy_lines,
                                                 force_lines,
                                                 pos_lines):
        energies.append(float(lines[energy_line + 1].strip()))

        force_curr = []
        for comp_line in lines[force_line+1:force_line + noa]:
            force_curr.append([float(n) for n in comp_line[2:-2].split()])
        force_curr.\
            append([float(n) for n in lines[force_line + noa][2:-3].split()])
        force_curr = np.array(force_curr)
        forces.append(force_curr)

        pos_curr = []
        for comp_line in lines[pos_line+1:pos_line + noa]:
            pos_curr.append([float(n) for n in comp_line[2:-2].split()])
        pos_curr.\
            append([float(n) for n in lines[pos_line + noa][2:-3].split()])
        pos_curr = np.array(pos_curr)
        positions.append(pos_curr)



    return energies, forces, positions
