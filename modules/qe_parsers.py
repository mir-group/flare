import numpy as np

def parse_md_output(outfile):

    steps = {}

    with open(outfile, 'r') as outf:
        lines = outf.readlines()

    split_indexes = [N for N in range(len(lines)) if '!' == lines[N][0]]

    step_chunks = []
    for n in range(len(split_indexes)):
        step_chunks.append(lines[split_indexes[n]:split_indexes[n+1] if
                           n != len(split_indexes)-1 else len(lines)])

    for current_chunk in step_chunks:

        force_start_line = [line for line in current_chunk if
                            'Forces acting on atoms' in line][0]
        force_end_line = [line for line in current_chunk if
                          'Total force' in line][0]
        force_start_index = current_chunk.index(force_start_line)+2
        force_end_index = current_chunk.index(force_end_line)-2

        atoms_start_line = [line for line in current_chunk if
                            'ATOMIC_POSITIONS' in line][0]
        atoms_end_line = [line for line in current_chunk if
                          'kinetic energy' in line][0]
        atoms_start_index = current_chunk.index(atoms_start_line)+1
        atoms_end_index = current_chunk.index(atoms_end_line)-3

        temperature_line = [line for line in current_chunk if
                            'temperature' in line][0]
        dyn_line = [line for line in current_chunk if
                    'Entering Dynamics' in line][0]
        dyn_index = current_chunk.index(dyn_line)
        time_index = dyn_index+1

        forces = []
        for line in current_chunk[force_start_index:force_end_index+1]:
            forceline = line.split('=')[-1].split()
            forces.append(np.array([float(forceline[0]), float(forceline[1]),
                          float(forceline[2])]))
        total_force = float(force_end_line.split('=')[1].strip().split()[0])
        SCF_corr = float(force_end_line.split('=')[2].strip()[0])

        positions = []
        elements = []
        for line in current_chunk[atoms_start_index:atoms_end_index+1]:
            atomline = line.split()
            elements.append(atomline[0])
            positions.append(np.array([float(atomline[1]), float(atomline[2]),
                             float(atomline[3])]))

        toten = float(current_chunk[0].split('=')[-1].strip().split()[0])
        temperature_line = temperature_line.split('=')[-1]
        temperature = float(temperature_line.split()[0])
        iteration = int(dyn_line.split('=')[-1])
        timeline = current_chunk[time_index].split('=')[-1].strip().split()[0]
        time = float(timeline)
        Ekin = float(atoms_end_line.split('=')[1].strip().split()[0])

        steps[iteration] = {'iteration': iteration,
                            'forces': forces,
                            'positions': positions,
                            'elements': elements,
                            'temperature': temperature,
                            'time': time,
                            'energy': toten,
                            'ekin': Ekin,
                            'kinetic energy': Ekin,
                            'total energy': toten,
                            'total force': total_force,
                            'SCF correction': SCF_corr}

    return(steps)
