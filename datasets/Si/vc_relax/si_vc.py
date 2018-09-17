import numpy as np
import sys
import os
sys.path.append('../../../modules')
import qe_input

input_file_name = './si_vc.in'
output_file_name = './si_vc.out'
pw_loc = os.environ.get('PWSCF_COMMAND')
calculation = 'vc-relax'

# specify diamnond crystal structure
cp = 2.69
cell = np.array([[0, cp, cp],
                 [cp, 0, cp],
                 [cp, cp, 0]])
positions = [np.array([0, 0, 0]),
             np.array([cp/2, cp/2, cp/2])]


scf_inputs = dict(pseudo_dir=os.environ.get('ESPRESSO_PSEUDO'),
                  outdir='./output',
                  nat=2,
                  ntyp=1,
                  ecutwfc=18.0,
                  ecutrho = 18.0 * 4
                  cell=cell,
                  species=['Si', 'Si'],
                  positions=positions,
                  kvec=np.array([4, 4, 4]),
                  ion_names=['Si'],
                  ion_masses=[28.086],
                  ion_pseudo=['Si.pz-vbc.UPF'])

si_vc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                         calculation, scf_inputs)

si_vc.run_espresso()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')
