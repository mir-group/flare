import numpy as np
import otf_parser_v2
import analyze_md
import kernels
import otf
import env
import concurrent.futures
import time


class TestOTF:
    def __init__(self, otf_file, gp_cell, kernel,
                 kernel_grad, cutoffs, start_list, test_snaps):
        self.test_snaps = test_snaps
        self.struc = None  # populate later

        # make gp and aimd run
        otf_run = otf_parser_v2.OtfAnalysis(otf_file)
        call_no = len(otf_run.gp_position_list)
        self.gp_model = otf_run.make_gp(gp_cell, kernel, kernel_grad, 'BFGS',
                                        call_no, start_list, cutoffs)

    # adapted from analyze_gp.py
    def predict_forces_on_test_set(self, aimd_file, aimd_cell):
        # get aimd trajectory
        aimd_run = analyze_md.MDAnalysis(aimd_file, aimd_cell)

        all_predictions = np.array([])
        all_stds = np.array([])
        all_forces = np.array([])

        for snap in self.test_snaps:
            # get structure and forces from AIMD trajectory
            self.struc = aimd_run.get_structure_from_snap(snap)
            forces_curr = aimd_run.get_forces_from_snap(snap)

            # predict forces and stds
            self.predict_on_structure_par()
            print(time.time())
            predictions = self.struc.forces
            stds = self.struc.stds

            # append results
            all_predictions = np.append(all_predictions, predictions)
            all_stds = np.append(all_stds, stds)
            all_forces = np.append(all_forces, forces_curr)

        return all_predictions, all_stds, all_forces

    def predict_on_structure_par(self):
        atom_list = list(range(self.struc.positions.shape[0]))
        n = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.predict_on_atom, atom_list):
                for i in range(3):
                    self.struc.forces[n][i] = res[0][i]
                    self.struc.stds[n][i] = res[1][i]
                n += 1

    def predict_on_structure(self):
        for n in range(self.struc.nat):
            chemenv = \
                env.AtomicEnvironment(self.struc, n, self.gp_model.cutoffs)
            for i in range(3):
                force, var = self.gp_model.predict(chemenv, i + 1)
                self.struc.forces[n][i] = float(force)
                self.struc.stds[n][i] = np.sqrt(np.absolute(var))

    # take prediction functions from otf
    def predict_on_atom(self, atom):
        chemenv = \
            env.AtomicEnvironment(self.struc, atom, self.gp_model.cutoffs)
        comps = []
        stds = []
        # predict force components and standard deviations
        for i in range(3):
            force, var = self.gp_model.predict(chemenv, i+1)
            comps.append(float(force))
            stds.append(np.sqrt(np.absolute(var)))

        return comps, stds
