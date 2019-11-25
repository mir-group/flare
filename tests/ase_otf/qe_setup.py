import os
from ase.calculators.espresso import Espresso

# set up executable
label = 'C'
input_file_name = label+'.pwi'
output_file_name = label+'.pwo'
no_cpus = 32
npool = 32
pw_loc = os.environ.get('PWSCF_COMMAND')
#pw_loc = '/n/home08/xiey/q-e/bin/pw.x'
os.environ['ASE_ESPRESSO_COMMAND'] = 'srun -n {0} --mpi=pmi2 {1} -npool {2} < {3} > {4}'.format(no_cpus, 
                            pw_loc, npool, input_file_name, output_file_name)
#os.environ['ASE_ESPRESSO_COMMAND'] = '{0} < {1} > {2}'.format(pw_loc, input_file_name, output_file_name)

# set up input parameters
input_data = {'control':   {'prefix': label, 
                            'pseudo_dir': './',
                            'outdir': './out',
                            #'verbosity': 'high',
                            'calculation': 'scf'},
              'system':    {'ibrav': 0, 
                            'ecutwfc': 60,
                            'ecutrho': 360},
              'electrons': {'conv_thr': 1.0e-9,
                            #'startingwfc': 'file',
                            'electron_maxstep': 100,
                            'mixing_beta': 0.7}}

# pseudo-potentials              
ion_pseudo = {'C': 'C.pz-rrkjus.UPF'}

# create ASE calculator
dft_calc = Espresso(pseudopotentials=ion_pseudo, label=label, 
                    tstress=True, tprnfor=True, nosym=True, #noinv=True,
                    input_data=input_data, kpts=(8, 8, 8)) 

