import os
from ase.calculators.espresso import Espresso

# set up executable
label = 'AgI'
input_file_name = label+'.pwi'
output_file_name = label+'.pwo'
no_cpus = 1
npool = 1
pw_loc = os.environ.get('PWSCF_COMMAND')
#pw_loc = '/n/home08/xiey/q-e/bin/pw.x'
#os.environ['ASE_ESPRESSO_COMMAND'] = 'srun -n {0} --mpi=pmi2 {1} -npool {2} < {3} > {4}'.format(no_cpus, 
#                            pw_loc, npool, input_file_name, output_file_name)
os.environ['ASE_ESPRESSO_COMMAND'] = '{0} < {1} > {2}'.format(pw_loc, input_file_name, output_file_name)

# set up input parameters
input_data = {'control':   {'prefix': label, 
                            'pseudo_dir': 'test_files/pseudos/',
                            'outdir': './out',
                            #'verbosity': 'high',
                            'calculation': 'scf'},
              'system':    {'ibrav': 0, 
                            'ecutwfc': 20, # 45,
                            'ecutrho': 40, # 181,
                            'smearing': 'gauss',
                            'degauss': 0.02,
                            'occupations': 'smearing'},
              'electrons': {'conv_thr': 1.0e-02,
                            #'startingwfc': 'file',
                            'electron_maxstep': 100,
                            'mixing_beta': 0.7}}

# pseudo-potentials              
ion_pseudo = {'Ag': 'Ag.pbe-n-kjpaw_psl.1.0.0.UPF', 
              'I':  'I.pbe-n-kjpaw_psl.1.0.0.UPF'}

# create ASE calculator
dft_calc = Espresso(pseudopotentials=ion_pseudo, label=label, 
                    tstress=True, tprnfor=True, nosym=True, #noinv=True,
                    input_data=input_data, kpts=(1,1,1)) 
