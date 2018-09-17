import numpy as np
import sys
import os
sys.path.append('../../../modules')
import qe_input

input_file_name = './C_vc.in'
output_file_name = './C_vc.out'
pw_loc = os.environ.get('PWSCF_COMMAND')
calculation = 'scf'

# specify diamnond crystal structure
cp = 1.763
# cp = 1.641877323
cell = np.array([[0, cp, cp],
                 [cp, 0, cp],
                 [cp, cp, 0]])
positions = [np.array([0, 0, 0]),
             np.array([cp/2, cp/2, cp/2])]

# set convergence parameters
nk = 30
ecutwfc = 75
ecutrho = 8 * ecutwfc


scf_inputs = dict(pseudo_dir=os.environ.get('ESPRESSO_PSEUDO'),
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
                  ion_masses=[12.011],
                  ion_pseudo=['C.pz-rrkjus.UPF'])

scf = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                       calculation, scf_inputs)

scf.run_espresso()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')
