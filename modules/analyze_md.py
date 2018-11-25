import numpy as np
import qe_parsers
import sys
sys.path.append('../../otf/otf_engine')
import env, gp, struc, kernels, qe_parsers


class MDAnalysis:
    def __init__(self, MD_output_file: str, cell: np.ndarray):
        self.MD_output_file = MD_output_file
        self.cell = cell
        self.MD_data = self.get_data_from_file()

    def get_data_from_file(self):
        data = qe_parsers.parse_md_output(self.MD_output_file)
        return data

    def get_MSD(self, skip):
        times = []
        MSDs = []
        init_pos = self.MD_data[1]['positions']
        no_snaps = len(self.MD_data)
        for n in np.arange(0, no_snaps, skip):
            time = self.MD_data[n+1]['time']
            positions = self.MD_data[n+1]['positions']
            disps = []
            for count, pos in enumerate(positions):
                disp = np.linalg.norm(pos - init_pos[count])
                sq_disp = disp**2
                disps.append(sq_disp)
            MSD = np.mean(disps)
            times.append(time)
            MSDs.append(MSD)
        return times, MSDs

    def get_MSD_of_atom(self, skip, atom):
        times = []
        MSDs = []
        init_pos = self.MD_data[1]['positions']
        no_snaps = len(self.MD_data)
        for n in np.arange(0, no_snaps, skip):
            time = self.MD_data[n+1]['time']
            positions = self.MD_data[n+1]['positions']
            pos = positions[atom]
            disp = np.linalg.norm(pos - init_pos[atom])
            sq_disp = disp**2
            times.append(time)
            MSDs.append(sq_disp)
        return times, MSDs

    def get_temperature(self, skip):
        times = []
        temps = []
        no_snaps = len(self.MD_data)
        for n in np.arange(0, no_snaps, skip):
            times.append(self.MD_data[n+1]['time'])
            temps.append(self.MD_data[n+1]['temperature'])
        return times, temps

    def get_structure_from_snap(self, snap, cutoff):
        positions = self.MD_data[snap]['positions']
        species = self.MD_data[snap]['elements']
        structure = struc.Structure(self.cell, species, positions,
                                    cutoff)
        return structure

    def get_forces_from_snap(self, snap):
        forces = self.MD_data[snap+1]['forces']
        return forces
