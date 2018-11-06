import numpy as np
from matplotlib import pyplot as plt


def read_pos(filename):
    pos = []

    with open(filename, 'r') as outf:
        lines = outf.readlines()

    for line in lines:
        line = line.split()

        curr_pos = np.zeros(shape=(3,))
        curr_pos[0] = str(line[1])
        curr_pos[1] = str(line[2])
        curr_pos[2] = str(line[3])
        pos.append(curr_pos)

    return np.array(pos)


def get_v(pos, prev_pos, next_pos):
    v = np.zeros(shape=pos.shape)

    for i in range(pos.shape[0]):
        v[i, :] = (next_pos[i, :] - prev_pos[i, :]) / (2 * dt)

    return v


def get_temp(pos, prev_pos, next_pos):
    prev_positions_m = prev_pos * angst2m
    positions_m = pos * angst2m
    next_positions_m = next_pos * angst2m

    v = get_v(pos=positions_m, prev_pos=prev_positions_m, next_pos=next_positions_m)
    v_magn = np.empty(v.shape[0])

    for i in range(v_magn.shape[0]):
        v_magn[i] = np.sqrt(v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2)

    T = (1/((3 * n_atoms - 3) * k_b)) * np.sum((v_magn**2) * m_al)

    return T


def parse_pos_otf(filename):
    traj = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    n_steps = 0

    for index, line in enumerate(lines):

        if line.startswith("- Frame "):
            n_steps += 1
            pos = []

            for frame_line in lines[(index+2):(index+2+n_atoms)]:
                frame_line = frame_line.split()
                curr_pos = np.zeros(shape=(3,))
                curr_pos[0] = str(frame_line[1])
                curr_pos[1] = str(frame_line[2])
                curr_pos[2] = str(frame_line[3])

                pos.append(curr_pos)

            traj.append(np.array(pos))

        if n_steps >= n_comp:
            return traj

    return traj

if __name__ == "__main__":

    otf_filename = '/Users/simonbatzner1/Desktop/Research/Research_Code/otf/datasets/Al_vac/pc_wp_Al_vac/otf_run.out'

    # params
    n_atoms = 31
    n_comp = 5000
    dt = 1e-15                                          # sec
    m_al = (26.98154 * 0.001) / (6.022140857 * 1e23)    # kg/atom
    k_b = 1.38064852 * 1e-23                            # J/K
    angst2m = 1e-10                                     # Angstrom to m

    # get otf temperature
    temps_otf = []
    pos_traj = parse_pos_otf(filename=otf_filename)

    for i in range(len(pos_traj)-2):
        temps_otf.append(get_temp(pos=pos_traj[i+1], prev_pos=pos_traj[i], next_pos=pos_traj[i+2]))

    # plot
    plt.plot(list(range(len(pos_traj)-2)), temps_otf)
    plt.xlabel('Frame')
    plt.ylabel('Instantaneous Temperature [K]')
    plt.show()

    np.savetxt('temps_otf.txt', np.array(temps_otf))

