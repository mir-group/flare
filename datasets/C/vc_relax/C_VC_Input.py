import numpy as np
import sys
import os
sys.path.append('../../../modules')
import qe_input

input_file_name = './C_vc.in'
output_file_name = './C_vc.out'
pw_loc = os.environ.get('PWSCF_COMMAND')
calculation = 'vc-relax'

# specify diamnond crystal structure
cp = 1.76
# cp = 1.641877323
cell = np.array([[0, cp, cp],
                 [cp, 0, cp],
                 [cp, cp, 0]])
positions = [np.array([0, 0, 0]),
             np.array([cp/2, cp/2, cp/2])]
nk = 20


scf_inputs = dict(pseudo_dir=os.environ.get('ESPRESSO_PSEUDO'),
                  outdir='./output',
                  nat=2,
                  ntyp=1,
                  ecutwfc=50.0,
                  ecutrho=200.0,
                  cell=cell,
                  species=['C', 'C'],
                  positions=positions,
                  kvec=np.array([nk, nk, nk]),
                  ion_names=['C'],
                  ion_masses=[12.011],
                  ion_pseudo=['C.pz-rrkjus.UPF'])

C_vc = qe_input.QEInput(input_file_name, output_file_name, pw_loc,
                        calculation, scf_inputs,
                        press_conv_thr='0.5D0')

C_vc.run_espresso()

# remove output directory
if os.path.isdir('output'):
    os.system('rm -r output')
