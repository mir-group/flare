# MD_Parser

def parse_qe_pwscf_md_output(outfile):
#def parse_qe_pwscf_md_output(path):

    steps={}

    # Get the lines out of the file first
    with open(outfile, 'r') as outf:
        lines = outf.readlines()

    # Because every step is marked by a total energy printing with the !
    # as the first character of the line, partition the file of output
    # into all different chunks of run data

    # Get the indexes to carve up the document later
    split_indexes=[N for N in range(len(lines)) if '!'==lines[N][0]]

    # Cut out the first chunk 
    # TODO: Analyze first chunk
    first_chunk=lines[0:split_indexes[0]]

    step_chunks = []
    # Carve up into chunks
    for n in range(len(split_indexes)):
        step_chunks.append(lines[split_indexes[n]:split_indexes[n+1] if n!=len(split_indexes)-1 else len(lines)]) 



    # Iterate through chunks
    for current_chunk in step_chunks:


        # Iterate through to find the bounds of regions of interest

        # Forces
        force_start_line = [line for line in current_chunk if 'Forces acting on atoms' in line][0]
        force_end_line   = [line for line in current_chunk if 'Total force' in line][0]
        force_start_index = current_chunk.index(force_start_line)+2
        force_end_index = current_chunk.index(force_end_line)-2

        # Positions
        atoms_start_line = [line for line in current_chunk if 'ATOMIC_POSITIONS' in line][0]
        atoms_end_line   = [line for line in current_chunk if 'kinetic energy' in line][0]
        atoms_start_index = current_chunk.index(atoms_start_line)+1
        atoms_end_index = current_chunk.index(atoms_end_line)-3

        # Misc Facts
        temperature_line = [ line for line in current_chunk if 'temperature' in line][0]
        dyn_line = [line for line in current_chunk if 'Entering Dynamics' in line][0]
        dyn_index = current_chunk.index(dyn_line)
        time_index = dyn_index+1

        # Parse through said regions of interest to get the information out

        forces = []
        for line in current_chunk[force_start_index:force_end_index+1]:
            forceline= line.split('=')[-1].split()
            forces.append([float(forceline[0]),float(forceline[1]),float(forceline[2])])
        total_force = float(force_end_line.split('=')[1].strip().split()[0])
        SCF_corr    = float(force_end_line.split('=')[2].strip()[0])


        positions =[]
        elements=[]
        for line in current_chunk[atoms_start_index:atoms_end_index+1]:
            atomline = line.split()
            elements.append(atomline[0])
            positions.append([float(atomline[1]),float(atomline[2]),float(atomline[3])])

        # Get Misc info 
        toten = float(current_chunk[0].split('=')[-1].strip().split()[0])
        temperature_line = temperature_line.split('=')[-1]
        temperature = float(temperature_line.split()[0])
        iteration = int(dyn_line.split('=')[-1])
        timeline = current_chunk[time_index].split('=')[-1].strip().split()[0]
        time = float( timeline)
        Ekin = float(atoms_end_line.split('=')[1].strip().split()[0])


        # Record the data associated with this step
        steps[iteration]={'iteration':iteration,
                           'forces':forces, 
                           'positions':positions,
                           'elements':elements,
                           'temperature':temperature,
                           'time':time,
                           'energy':toten,
                           'ekin':Ekin,
                           'kinetic energy':Ekin,
                           'total energy':toten,
                           'total force':total_force,
                           'SCF correction':SCF_corr}

    return(steps)