import sys
import numpy as np
import crystals
sys.path.append('../otf_engine')
import gp, env, struc, kernels, otf


class OtfAnalysis:
    def __init__(self, filename):
        self.filename = filename

        positions, dft_frames, temperatures, times, msds, dft_times = \
            self.parse_pos_otf(filename)

        self.positions = positions
        self.dft_frames = dft_frames
        self.temperatures = temperatures
        self.times = times
        self.msds = msds
        self.dft_times = dft_times

        gp_pos_list, gp_force_list, gp_atom_list, gp_hyp_list, \
            gp_cutoff_radius, gp_species_list = self.extract_gp_info(filename)

        self.gp_pos_list = gp_pos_list
        self.gp_force_list = gp_force_list
        self.gp_atom_list = gp_atom_list
        self.gp_hyp_list = gp_hyp_list
        self.gp_cutoff_radius = gp_cutoff_radius
        self.gp_species_list = gp_species_list

    def make_gp(self, cell, kernel, bodies, algo, call_no):
        gp_model = gp.GaussianProcess(kernel, bodies, algo)
        gp_hyps = self.gp_hyp_list[call_no-1]
        gp_model.hyps = gp_hyps

        for count, (positions, forces, atom, _, species) in \
            enumerate(zip(self.gp_pos_list, self.gp_force_list,
                          self.gp_atom_list, self.gp_hyp_list,
                          self.gp_species_list)):
            if count < call_no:
                struc_curr = struc.Structure(cell, species, positions,
                                             self.gp_cutoff_radius)

                gp_model.update_db(struc_curr, forces, custom_range=[atom])

        gp_model.set_L_alpha()

        return gp_model

    @staticmethod
    def get_gp_activation(gp_model):
        pass

    def parse_pos_otf(self, filename):
        positions = []
        temperatures = []
        dft_frames = []
        dft_times = []
        times = []
        msds = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        n_steps = 0

        for index, line in enumerate(lines):
            if line.startswith("Number of Atoms"):
                at_line = line.split()
                noa = int(at_line[3])

            if line.startswith("-*"):
                dft_line = line.split()
                dft_frames.append(int(dft_line[1]))
                dft_times.append(float(dft_line[4]))

            if line.startswith("- Frame"):
                n_steps += 1
                pos = []

                frame_line = line.split()
                sim_time = float(frame_line[5])
                times.append(sim_time)

                for frame_line in lines[(index+2):(index+2+noa)]:
                    frame_line = frame_line.split()
                    curr_pos = np.zeros(shape=(3,))
                    curr_pos[0] = str(frame_line[1])
                    curr_pos[1] = str(frame_line[2])
                    curr_pos[2] = str(frame_line[3])

                    pos.append(curr_pos)

                temp_line = lines[index+2+noa].split()
                temperatures.append(float(temp_line[1]))

                positions.append(np.array(pos))

                msds.append(np.mean((pos - positions[0])**2))

        return positions, dft_frames, temperatures, times, msds, dft_times

    def extract_gp_info(self, filename):
        pos_list = []
        species_list = []
        atom_list = []
        force_list = []
        hyp_list = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        for index, line in enumerate(lines):
            if line.startswith("Structure Cutoff Radius:"):
                line_curr = line.split()
                cutoff_radius = float(line_curr[3])

            if line.startswith("Number of Atoms"):
                line_curr = line.split()
                noa = int(line_curr[3])

            if line.startswith("# Hyperparameters"):
                line_curr = line.split()
                noh = int(line_curr[2])

            if line.startswith("Calling DFT with training atoms"):
                line_curr = line.split()

                pos = []
                fcs = []
                hyps = []
                species = []

                for frame_line in lines[(index+4):(index+4+noh)]:
                    frame_line = frame_line.split()
                    hyps.append(float(frame_line[5]))
                hyps = np.array(hyps)
                hyp_list.append(hyps)
                hyp_list.append(hyps)

                for frame_line in lines[(index+12):(index+12+noa)]:
                    frame_line = frame_line.split()
                    curr_pos = np.zeros(shape=(3,))
                    curr_pos[0] = str(frame_line[1])
                    curr_pos[1] = str(frame_line[2])
                    curr_pos[2] = str(frame_line[3])

                    dft_fc = np.zeros(shape=(3,))
                    dft_fc[0] = str(frame_line[4])
                    dft_fc[1] = str(frame_line[5])
                    dft_fc[2] = str(frame_line[6])

                    species.append(str(frame_line[0]))
                    pos.append(curr_pos)
                    fcs.append(dft_fc)

                pos = np.array(pos)
                pos_list.append(pos)
                pos_list.append(pos)

                species_list.append(species)
                species_list.append(species)

                fcs = np.array(fcs)
                force_list.append(fcs)
                force_list.append(fcs)

                atom_list.append(0)
                atom_list.append(30)

            if line.startswith("Calling DFT due to"):
                line_curr = line.split()

                pos = []
                fcs = []
                hyps = []
                species = []

                for frame_line in lines[(index+4):(index+4+noh)]:
                    frame_line = frame_line.split()
                    hyps.append(float(frame_line[5]))
                hyps = np.array(hyps)
                hyp_list.append(hyps)

                for frame_line in lines[(index+12):(index+12+noa)]:
                    frame_line = frame_line.split()
                    curr_pos = np.zeros(shape=(3,))
                    curr_pos[0] = str(frame_line[1])
                    curr_pos[1] = str(frame_line[2])
                    curr_pos[2] = str(frame_line[3])

                    dft_fc = np.zeros(shape=(3,))
                    dft_fc[0] = str(frame_line[4])
                    dft_fc[1] = str(frame_line[5])
                    dft_fc[2] = str(frame_line[6])

                    pos.append(curr_pos)
                    fcs.append(dft_fc)

                    species.append(str(frame_line[0]))

                pos = np.array(pos)
                pos_list.append(pos)

                fcs = np.array(fcs)
                force_list.append(fcs)

                species_list.append(species)

                atom_list.append(int(line_curr[5]))

        return pos_list, force_list, atom_list, hyp_list, cutoff_radius,\
            species_list


def get_gp_from_file(filename, cell):
    pass
