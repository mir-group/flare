import numpy as np
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

    def get_structure_from_snap(self, snap, cutoff):
        positions = self.MD_data[snap]['positions']
        species = self.MD_data[snap]['elements']
        structure = struc.Structure(self.cell, species, positions,
                                    cutoff)
        return structure

    def get_forces_from_snap(self, snap):
        forces = self.MD_data[snap+1]['forces']
        return forces


# return untrained gp model
def get_gp_from_snaps(md_trajectory, training_snaps, cutoff, kernel, bodies):
    gp_model = gp.GaussianProcess(kernel, bodies)

    for snap in training_snaps:
        structure = md_trajectory.get_structure_from_snap(snap, cutoff)
        forces = md_trajectory.get_forces_from_snap(snap)
        gp_model.update_db(structure, forces)

    return gp_model


def predict_forces_on_structure(gp_model, md_trajectory, snap, cutoff):
    structure = md_trajectory.get_structure_from_snap(snap, cutoff)
    forces = np.array(md_trajectory.get_forces_from_snap(snap))
    noa = len(structure.positions)

    predictions = np.zeros([noa, 3])
    variances = np.zeros([noa, 3])

    for atom in range(noa):
        env_curr = env.ChemicalEnvironment(structure, atom)

        for n in range(3):
            d = n+1
            comp_pred, comp_var = gp_model.predict(env_curr, d)
            predictions[atom, n] = comp_pred
            variances[atom, n] = comp_var

    # mean absolute error
    MAE = np.mean(np.abs(forces-predictions))

    # mean absolute std
    stds = np.sqrt(variances)
    MAS = np.mean(stds)

    return predictions, variances, forces, MAE, MAS


def predict_forces_on_test_set(gp_model, md_trajectory, snaps, cutoff):
    all_predictions = []
    all_variances = []
    all_forces = []
    for snap in snaps:
        predictions, variances, forces, _, _ = \
            predict_forces_on_structure(gp_model, md_trajectory, snap, cutoff)
        predictions = predictions.reshape(predictions.size)
        variances = variances.reshape(variances.size)
        forces = forces.reshape(forces.size)

        for m in range(len(predictions)):
            all_predictions.append(predictions[m])
            all_variances.append(variances[m])
            all_forces.append(forces[m])

    all_predictions = np.array(all_predictions)
    all_variances = np.array(all_variances)
    all_forces = np.array(all_forces)

    return all_predictions, all_variances, all_forces




if __name__ == '__main__':
    # define md trajectory object
    print('making md object...')
    C_data_file = \
        '/Users/jonpvandermause/Research/GP/otf/datasets/C/constant_T/C.out'
    cube_lat = 2 * 1.763391008
    cell = np.eye(3) * cube_lat

    md_trajectory = MDAnalysis(C_data_file, cell)

    # make example gp
    print('training gp model...')
    training_snaps = [200]
    cutoff = 3
    kernel = 'n_body_sc'
    bodies = 3

    gp_test = \
        get_gp_from_snaps(md_trajectory, training_snaps, cutoff, kernel,
                          bodies)
    gp_test.train(True)

    # make prediction
    print('making prediction...')
    test_snap = 500

    predictions, variances, forces, MAE, MAS = \
        predict_forces_on_structure(gp_test, md_trajectory, test_snap, cutoff)

    print(gp_test.hyps)
    print('mean absolute std: %.6f Ryd/Bohr' % MAS)
    print('mean absolute error: %.6f Ryd/Bohr' % MAE)
