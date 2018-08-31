import numpy as np
import scipy as sp
import scipy.linalg
import time
from scipy.optimize import minimize
import multiprocessing as mp
import math


# -----------------------------------------
#				environment
# -----------------------------------------

# given list of Cartesian coordinates, return list of atomic environments
def get_cutoff_vecs(vec, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    # get bravais coefficients
    coeff = np.matmul(brav_inv, vec)
    
    # get bravais coefficients for atoms within one super-super-cell
    coeffs = [[],[],[]]
    for n in range(3):
        coeffs[n].append(coeff[n])
        coeffs[n].append(coeff[n]-1)
        coeffs[n].append(coeff[n]+1)
        coeffs[n].append(coeff[n]-2)
        coeffs[n].append(coeff[n]+2)

    # get vectors within cutoff
    vecs = []
    dists = []
    for m in range(len(coeffs[0])):
        for n in range(len(coeffs[1])):
            for p in range(len(coeffs[2])):
                vec_curr = coeffs[0][m]*vec1 + coeffs[1][n]*vec2 + coeffs[2][p]*vec3
                
                dist = np.linalg.norm(vec_curr)

                if dist < cutoff:
                    vecs.append(vec_curr)
                    dists.append(dist)
                    
    return vecs, dists

# ordering convention: atoms (a,b,c) correspond to distances (ab, ac, bc)
def order_triplet(c,r1,t1,x1,y1,z1,r2,t2,x2,y2,z2):
    # calculate third distance
    r3 = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
    
    labs_init = [c,t1,t2]
    dists_init = [r1,r2,r3]
    xs_init = [x1,x2,0]
    ys_init = [y1,y2,0]
    zs_init = [z1,z2,0]
    xrel_init = [x1/r1,x2/r2,0]
    yrel_init = [y1/r1,y2/r2,0]
    zrel_init = [z1/r1,z2/r2,0]
    
    # order the labels
    # all atoms the same
    if (c==t1) and (c==t2):
        trip_type = 1
        atom_order = [0,1,2]
        dist_order = [0,1,2]
    # two alike, one different: put different atom first
    if (c==t1) and (c!=t2):
        trip_type = 2
        atom_order = [2,0,1]
        dist_order = [1,2,0]
    if (c==t2) and (c!=t1):
        trip_type = 2
        atom_order = [1,2,0]
        dist_order = [2,0,1]
    if (t1==t2) and (c!=t1):
        trip_type = 2
        atom_order = [0,1,2]
        dist_order = [0,1,2]
    # all atoms different: sort atom labels alphabetically
    if (c!=t1) and (c!=t2) and (t1!=t2):
        trip_type = 3
        atom_order = list(np.argsort(labs_init))
        # check all 6 possible orderings
        if inds[0]==0 and inds[1]==1 and inds[2]==2:
            dist_order = [0,1,2]
        if inds[0]==0 and inds[1]==2 and inds[2]==1:
            dist_order = [1,0,2]
        if inds[0]==1 and inds[1]==0 and inds[2]==2:
            dist_order = [0,2,1]
        if inds[0]==1 and inds[1]==2 and inds[2]==0:
            dist_order = [2,0,1]
        if inds[0]==2 and inds[1]==0 and inds[2]==1:
            dist_order = [1,2,0]
        if inds[0]==2 and inds[1]==1 and inds[2]==0:
            dist_order = [2,1,0]
            
    trip_labs = [labs_init[n] for n in atom_order]
    trip_dists = [dists_init[n] for n in dist_order]
    trip_xs = [xs_init[n] for n in dist_order]
    trip_ys = [ys_init[n] for n in dist_order]
    trip_zs = [zs_init[n] for n in dist_order]
    trip_xrel = [xrel_init[n] for n in dist_order]
    trip_yrel = [yrel_init[n] for n in dist_order]
    trip_zrel = [zrel_init[n] for n in dist_order]
    
    return trip_type, trip_labs, trip_dists, trip_xs, trip_ys, trip_zs, trip_xrel, trip_yrel, trip_zrel

# create triplet dictionary from 2-body dictionary
def get_trip_dict(tb_dict):
    trip_dict = {'typs':[],'labs':[],'dists':[],'xs':[],'ys':[],'zs':[], 'xrel':[], 'yrel':[],'zrel':[]}


    # pull relevant information from 2-body dictionary
    dist_no = len(tb_dict['dists'])
    c = tb_dict['central_atom']
    dists = tb_dict['dists']
    xs = tb_dict['xs']
    ys = tb_dict['ys']
    zs = tb_dict['zs']
    types = tb_dict['types']

    for m in range(dist_no):
        r1 = dists[m]
        t1 = types[m]
        x1 = xs[m]
        y1 = ys[m]
        z1 = zs[m]
        for n in range(m,dist_no):
            r2 = dists[n]
            t2 = types[n]
            x2 = xs[n]
            y2 = ys[n]
            z2 = zs[n]

            trip_type, trip_labs, trip_dists, trip_xs, trip_ys, trip_zs, trip_xrel, trip_yrel, trip_zrel =\
                order_triplet(c,r1,t1,x1,y1,z1,r2,t2,x2,y2,z2)

            # triplet already in dictionary:
            if trip_labs in trip_dict['labs']:
                lab_ind = trip_dict['labs'].index(trip_labs)
                trip_dict['dists'][lab_ind].append(trip_dists)
                trip_dict['xs'][lab_ind].append(trip_xs)
                trip_dict['ys'][lab_ind].append(trip_ys)
                trip_dict['zs'][lab_ind].append(trip_zs)
                trip_dict['xrel'][lab_ind].append(trip_xrel)
                trip_dict['yrel'][lab_ind].append(trip_yrel)
                trip_dict['zrel'][lab_ind].append(trip_zrel)
            # or else it needs to be appended:
            else:
                trip_dict['typs'].append(trip_type)
                trip_dict['labs'].append(trip_labs)
                trip_dict['dists'].append([trip_dists])
                trip_dict['xs'].append([trip_xs])
                trip_dict['ys'].append([trip_ys])
                trip_dict['zs'].append([trip_zs])
                trip_dict['xrel'].append([trip_xrel])
                trip_dict['yrel'].append([trip_yrel])
                trip_dict['zrel'].append([trip_zrel])
                
    return trip_dict

# given list of cartesian coordinates, get chemical environment of specified atom
# pos = list of cartesian coordinates
# typs = list of atom types
def get_env_struc(pos, typs, atom, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    pos_atom = np.array(pos[atom]).reshape(3,1)
    typ = typs[atom]
    env = {'central_atom':typ, 'dists':[],'xs':[],'ys':[],'zs':[],\
           'xrel':[],'yrel':[],'zrel':[],'types':[]}
    
    # loop through positions to find all atoms and images in the neighborhood
    for n in range(len(pos)):
        # position relative to reference atom
        diff_curr = np.array(pos[n]).reshape(3,1) - pos_atom

        # get images within cutoff
        vecs, dists = get_cutoff_vecs(diff_curr, brav_mat, \
            brav_inv, vec1, vec2, vec3, cutoff)

        for vec, dist in zip(vecs, dists):
            # ignore self interaction
            if dist != 0:
                # append distance
                env['dists'].append(dist)
                
                # append coordinate differences
                env['xs'].append(vec[0][0])
                env['ys'].append(vec[1][0])
                env['zs'].append(vec[2][0])
                
                # append relative coordinate differences
                env['xrel'].append(vec[0][0]/dist)
                env['yrel'].append(vec[1][0]/dist)
                env['zrel'].append(vec[2][0]/dist)
                
                # append atom type
                env['types'].append(typs[n])

    env['trip_dict']=get_trip_dict(env)
    
    return env

# given list of cartesian coordinates, return list of chemical environments
def get_envs(pos, typs, brav_mat, brav_inv, vec1, vec2, vec3, cutoff):
    envs = []
    for n in range(len(pos)):
        atom = n
        env = get_env_struc(pos, typs, atom, brav_mat, brav_inv, vec1, vec2, vec3, cutoff)
        envs.append(env)
        
    return envs

# -----------------------------------------
#				gp
# -----------------------------------------

# get 3Nx3N noiseless kernel matrix
# assume all 3 force components are known for each configuration
# X is assumed to be a list of environments
def get_K(X,sig,ls,noise,kernel):
    ds = ['xrel','yrel','zrel']
    
    # initialize matrix
    size = len(X)*3
    K = np.zeros([size, size])
    
    # calculate elements
    for m in range(size):
        x1 = X[int(math.floor(m/3))]
        d1 = ds[m%3]
        for n in range(m,size):
            x2 = X[int(math.floor(n/3))]
            d2 = ds[n%3]
           
            # calculate kernel
            cov = kernel(x1, x2, d1, d2, sig, ls)
            K[m,n] = cov
            K[n,m] = cov
    # perform cholesky decomposition
    L = np.linalg.cholesky(K+noise**2*np.eye(size))
    
    return K, L

def get_assignment(X,batches):
    ds = ['xrel','yrel','zrel']
    size = len(X)*3
    tot_comps = (size+1)*size/2
    switch = math.ceil(tot_comps/batches)
    batch_count = 0
    assign = []

    # calculate elements
    for m in range(size):
        x1_ind = int(math.floor(m/3))
        d1 = ds[m%3]
        for n in range(m,size):
            x2_ind = int(math.floor(n/3))
            d2 = ds[n%3]

            # when batch limit is reached, reset counter
            if batch_count == switch:
                batch_count = 0

            # if counter is zero, start a new list
            if batch_count==0:
                assign.append([[x1_ind,d1,x2_ind,d2]])

            # otherwise, add comparison to last list
            else:
                assign[len(assign)-1].append([x1_ind,d1,x2_ind,d2])

            # increment counter
            batch_count+=1

    return assign

def get_cov_list(pair_list, X, kernel, sig, ls):
    res_store = []
    for n in range(len(pair_list)):
        pair = pair_list[n]
        x1 = X[pair[0]]
        d1 = pair[1]
        x2 = X[pair[2]]
        d2 = pair[3]
        res_store.append(kernel(x1, x2, d1, d2, sig, ls))
    return res_store

def get_K_pool(X, batches, kernel, sig, ls, noise):
    # get assignment
    assign = get_assignment(X, batches)
    
    # create pool of processors
    procs = len(assign)
    pool = mp.Pool(processes = procs)

    # assign to processors
    res = []
    for n in range(procs):
        # set pair list
        pair_list = assign[n]

        # assign pair list to processor
        res.append(pool.apply_async(get_cov_list,\
            args = (pair_list, X, kernel, sig, ls)))

    # collect covariance lists
    results = []
    for n in range(procs):
        results.append(res[n].get())
        
    # create K matrix from covariance lists
    size = len(X)*3
    K = np.zeros([size, size])
    counter = 0
    cov_list = 0
    for m in range(size):
        for n in range(m,size):
            if counter < len(results[cov_list]):
                ent = results[cov_list][counter]
            else:
                counter = 0
                cov_list +=1
                ent = results[cov_list][counter]
            
            counter+=1
            K[m,n]=ent
            K[n,m]=ent

    # close the pool
    pool.close()
    
    # perform cholesky decomposition
    L = np.linalg.cholesky(K+noise**2*np.eye(size))
    
    return K, L

# get kernel vector
def get_kv(X,x,d1,sig,ls,kernel):
    ds = ['xrel','yrel','zrel']
    size = len(X)*3
    kv=np.zeros([size,1])
    for m in range(size):
        x2 = X[int(math.floor(m/3))]
        d2 = ds[m%3]
        kv[m]=kernel(x,x2,d1,d2,sig,ls)
        
    return kv

# get alpha
def get_alpha(K,L,y): 
    # get alpha
    ts1 = sp.linalg.solve_triangular(L,y,lower=True)
    alpha = sp.linalg.solve_triangular(L.transpose(),ts1)
            
    return alpha

# get likelihood
def get_like(K,L,y,alpha): 
    # get log marginal likelihood
    like = -(1/2)*np.matmul(y.transpose(),alpha)-\
            np.sum(np.log(np.diagonal(L)))-\
            np.log(2*np.pi)*K.shape[1]/2
            
    return like

# get likelihood as a function of hyperparameters
def like_hyp(hyp,X,y,kernel):
    # unpack hyperparameters
    sig = hyp[0]
    ls = hyp[1]
    noise = hyp[2]
    
    # calculate likelihood
    K, L = get_K(X,sig,ls,noise,kernel)
    alpha = get_alpha(K,L,y)
    like = get_like(K,L,y,alpha)
    
    return like

# get minus likelihood as a function of hyperparameters
def minus_like_hyp(hyp,X,y,kernel):
    like = like_hyp(hyp,X,y,kernel)
    minus_like = -like
    return minus_like

# make GP prediction with SE kernel
def GP_pred(X,K,L,alpha,sig,ls,xt,d,kernel):
    # get kernel vector
    kv = get_kv(X,xt,d,sig,ls,kernel)
    
    # get predictive mean
    f = np.matmul(kv.transpose(),alpha)
    
    # get predictive variance
    v = sp.linalg.solve_triangular(L,kv,lower=True)
    self_kern = kernel(xt, xt, d, d, sig, ls)
    var = self_kern - np.matmul(v.transpose(),v)
    
    return f, var

# convert list of triplets to column vector
def fc_conv(fcs):
    comp_len = len(fcs)*3
    comps = []
    for n in range(comp_len):
        fc_ind = int(math.floor(n/3))
        d = n%3
        comps.append(fcs[fc_ind][d])
    
    return np.array(comps).reshape(comp_len,1)

# -----------------------------------------
#				misc
# -----------------------------------------

# create text file
# inputs:
    # fname: name of created file
    # text: text of created file
def write_file(fname, text):
    with open(fname, 'w') as fin:
        fin.write(text)