import qe_input

# -----------------------------------------------------------------------------
#                  make bash file for running on odyssey
# -----------------------------------------------------------------------------

bash_name = 'data.sh'
command = """
source activate numba
python training_data.py"""  # tell slurm to run this script

bash_inputs = dict(n=32,
                   N=1,
                   t=14,
                   e='test.err',
                   p='kozinsky',
                   o='test.out',
                   mem_per_cpu=5000,
                   mail_user='jonathan_vandermause@g.harvard.edu',
                   command=command)

bash = qe_input.BashInput(bash_name, bash_inputs)
bash.write_bash_text()
