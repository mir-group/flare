import numpy as np
from convergence import convergence


input_file_name = './C_conv.in'
output_file_name = './C_conv.out'
pw_loc = '/n/holylfs/LABS/kozinsky_lab/Software/qe-6.2.1/bin/pw.x'
pseudo_dir = '/n/holylfs/LABS/kozinsky_lab/Software/qe-6.2.1/pseudo'
calculation = 'scf'

# specify diamond crystal structure
cp = 1.763
cell = np.array([[0, cp, cp],
                 [cp, 0, cp],
                 [cp, cp, 0]])
positions = [np.array([0, 0, 0]),
             np.array([cp/2, cp/2, cp/2])]

# set convergence parameters
nk = 40
ecutwfc = 100
ecutrho = 8 * ecutwfc


scf_inputs = dict(pseudo_dir=pseudo_dir,
                  outdir='./output',
                  nat=2,
                  ntyp=1,
                  ecutwfc=ecutwfc,
                  ecutrho=ecutrho,
                  cell=cell,
                  species=['C', 'C'],
                  positions=positions,
                  kvec=np.array([nk, nk, nk]),
                  ion_names=['C'],
                  ion_masses=[12],
                  ion_pseudo=['C.pz-rrkjus.UPF'])

# choose parameter grid
nks = [10, 30]
ecutwfcs = [20, 40]
rho_facs = [4, 8]

convergence(input_file_name, output_file_name, pw_loc,
            calculation, scf_inputs, nks, ecutwfcs, rho_facs)
