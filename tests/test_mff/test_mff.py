import numpy as np
import sys
sys.path.append('../../')
from flare import gp, env, struc, kernels
import flare.mff as mff
from flare.modules import qe_parsers, analyze_gp
import time

def compare(md_trajectory, mff_test, test_snaps, cutoffs, log_file, mean_only):
    '''
    :param data: file for testing
    :param test_snaps: snapshots selected for testing
    '''

    f = open(log_file, mode='w')

    mean_err = []
    var_err = []
    for s, snap in enumerate(test_snaps):
        structure = md_trajectory.get_structure_from_snap(snap)
        forces = md_trajectory.get_forces_from_snap(snap)
        atom_num = len(forces)
        
        f.write('\n'+'-'*20)
        f.write('snap:'+str(snap)+'\n')
        for atom in range(atom_num):
            t0 = time.time()
            atom_env = env.AtomicEnvironment(structure, atom, cutoffs)
            env_time = time.time()-t0
            atom_force = forces[atom]

            # ----- gp prediction --------
            t0 = time.time()
            gp_pred_x = mff_test.GP.predict(atom_env, 1)
            gp_pred_y = mff_test.GP.predict(atom_env, 2)
            gp_pred_z = mff_test.GP.predict(atom_env, 3)
            gp_pred = [(gp_pred_x[0], gp_pred_y[0], gp_pred_z[0]), \
                        (gp_pred_x[1], gp_pred_y[1], gp_pred_z[1])]                
            gp_time = time.time()-t0

            # ------ mff prediction ---------
            t0 = time.time()
            mff_pred = mff_test.predict(atom_env, mean_only)
            mff_time = time.time()-t0
            
            mean_err.append(np.absolute(np.array(gp_pred[0])-np.array(mff_pred[0])))
            var_err.append(np.absolute(np.array(gp_pred[1])-np.array(mff_pred[1])))

            f.write('true force:'+np.array2string(atom_force)+'\n')
            f.write('env time:'+str(env_time)+'\n')
            f.write('gp mean:'+str(gp_pred[0])+'\n')
            f.write('gp var:'+str(gp_pred[1])+'\n')
            f.write('gp time:'+str(gp_time)+'\n')
            f.write('mff mean:'+np.array2string(mff_pred[0])+'\n')
            f.write('mff var:'+np.array2string(mff_pred[1])+'\n')
            f.write('mff time:'+str(mff_time)+'\n')
            f.write('\n')

        f.write('\n')
    f.close()
    return np.mean(mean_err), np.mean(var_err)


def get_gp_from_snaps(md_trajectory, training_snaps, kernel, custom_range,
                      kernel_grad, hyps, cutoffs, algorithm='BFGS'):

    gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps,
                                  cutoffs, opt_algorithm=algorithm)

    for snap in training_snaps:
        structure = md_trajectory.get_structure_from_snap(snap)
        forces = md_trajectory.get_forces_from_snap(snap)
        gp_model.update_db(structure, forces, custom_range)
        gp_model.set_L_alpha()

    return gp_model

param_file = sys.argv[1]
param_dict = np.load(param_file).item()
GP_params = param_dict['GP_params']
grid_params = param_dict['grid_params']
struc_params = param_dict['struc_params']
data_file = 'Al_md.out' #param_dict['data_file']

struc_params['cube_lat'] *= np.eye(3)
print(GP_params, grid_params, struc_params)

cutoffs = GP_params['cutoffs']
kernel = GP_params['kernel']
kernel_grad = GP_params['kernel_grad']
hyps = GP_params['hyps']
hyp_labels = GP_params['hyp_labels']

species = struc_params['species']
bodies = grid_params['bodies']

# ------------build gp
def test(train_size, gn, mean_only):
    custom_range = [i for i in range(train_size)]
    cell = struc_params['cube_lat']
    md_trajectory = analyze_gp.MDAnalysis(data_file, cell)       

    t0 = time.time()
    training_snaps = [20, 50]
    print('traj built')
    GP = get_gp_from_snaps(md_trajectory, training_snaps, kernel, 
         custom_range, kernel_grad, hyps, cutoffs)
    #GP.train(monitor=False)
    GP.set_L_alpha() 
    print('gp building time:', time.time()-t0)     
    ky_mat_inv = GP.ky_mat_inv
    alpha = GP.alpha

    # test update_L_alpha
    snap = 20
    GP_new = get_gp_from_snaps(md_trajectory, [snap], kernel, 
         custom_range, kernel_grad, hyps, cutoffs)
    GP_new.set_L_alpha()
    snap = 50
    structure = md_trajectory.get_structure_from_snap(snap)
    forces = md_trajectory.get_forces_from_snap(snap)
    GP_new.update_db(structure, forces, custom_range)
    GP_new.update_L_alpha()
    new_ky_mat_inv = GP_new.ky_mat_inv
    new_alpha = GP_new.alpha
    print(np.all(np.absolute(new_ky_mat_inv - ky_mat_inv)<1e-3))
    print(np.all(np.absolute(new_alpha - alpha)<1e-3))

    raise('err')
    # test     
    test_snaps = [900]
    training_size = str(len(custom_range)*len(training_snaps))
    
    # mff params
    if bodies == 3:
        grid_params['grid_num_3'] = (gn, gn, gn)
        grid_params['svd_rank'] = np.min((3 * int(training_size), gn**3))
    elif bodies == 2:
        grid_params['grid_num_2'] = gn
    elif bodies == '2+3':
        grid_params['grid_num_3'] = (gn, gn, gn)
        grid_params['svd_rank'] = np.min((3 * int(training_size), gn**3))
        grid_params['grid_num_2'] = gn

    grid_number = str(gn)
    t0 = time.time()
    mff_test = mff.MappedForceField(GP, grid_params, struc_params)
    print('mff built, time:', time.time()-t0)
    print('svd rank:', mff_test.svd_rank)

    # save as a .txt file
    filelabel = species+'-g' + grid_number + '-t' + training_size + '-s'\
                + str(grid_params['svd_rank']) 
    
    filename = filelabel + '.txt'
    filepath = './'
    if mean_only:
        log_file = filepath+'mean-'+filename
    else:
        log_file = filepath + filename
    print('results saved in:', log_file)
    mean_err, var_err = compare(md_trajectory, mff_test, test_snaps, cutoffs, log_file, mean_only)
    return mean_err, var_err
    
max_num = int(sys.argv[2])
for train_size in [1]: #range(3, max_num+1):
    for g_power in [5]: #range(6):
#        gn = 2 ** (g_power+1)
        gn = 64
        mean_err, var_err = test(train_size, gn, mean_only=False)
        print('mean error:', mean_err)
        print('var error:', var_err)


