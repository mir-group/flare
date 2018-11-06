import numpy as np
import os


class BashInput:
    def __init__(self, bash_name, bash_inputs):
        self.bash_name = bash_name

        self.n = bash_inputs['n']
        self.N = bash_inputs['N']
        self.t = bash_inputs['t']
        self.e = bash_inputs['e']
        self.p = bash_inputs['p']
        self.o = bash_inputs['o']
        self.mem_per_cpu = bash_inputs['mem_per_cpu']
        self.mail_user = bash_inputs['mail_user']
        self.command = bash_inputs['command']

        self.bash_text = self.get_bash_text()

    def get_bash_text(self):
        sh_text = """#!/bin/sh
#SBATCH -n {}
#SBATCH -N {}
#SBATCH -t {}-00:00
#SBATCH -e {}
#SBATCH -p {}
#SBATCH -o {}
#SBATCH --mem-per-cpu={}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={}

module load gcc/4.9.3-fasrc01 openmpi/2.1.0-fasrc01
module load python/3.6.3-fasrc01

{}""".format(self.n, self.N, self.t, self.e, self.p, self.o,
             self.mem_per_cpu, self.mail_user, self.command)

        return sh_text

    def write_bash_text(self):
        with open(self.bash_name, 'w') as fin:
            fin.write(self.bash_text)


class QEInput:
    def __init__(self, input_file_name: str, output_file_name: str,
                 pw_loc: str, calculation: str,
                 scf_inputs: dict, md_inputs: dict = None,
                 press_conv_thr=None):

        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.pw_loc = pw_loc
        self.calculation = calculation

        self.pseudo_dir = scf_inputs['pseudo_dir']
        self.outdir = scf_inputs['outdir']
        self.nat = scf_inputs['nat']
        self.ntyp = scf_inputs['ntyp']
        self.ecutwfc = scf_inputs['ecutwfc']
        self.ecutrho = scf_inputs['ecutrho']
        self.cell = scf_inputs['cell']
        self.species = scf_inputs['species']
        self.positions = scf_inputs['positions']
        self.kvec = scf_inputs['kvec']
        self.ion_names = scf_inputs['ion_names']
        self.ion_masses = scf_inputs['ion_masses']
        self.ion_pseudo = scf_inputs['ion_pseudo']

        # get text blocks
        self.species_txt = self.get_species_txt()
        self.position_txt = self.get_position_txt()
        self.cell_txt = self.get_cell_txt()
        self.kpt_txt = self.get_kpt_txt()

        # get md parameters
        if self.calculation == 'md':
            self.dt = md_inputs['dt']
            self.nstep = md_inputs['nstep']
            self.ion_temperature = md_inputs['ion_temperature']
            self.tempw = md_inputs['tempw']

        # if vc, get pressure convergence threshold
        if self.calculation == 'vc-relax':
            self.press_conv_thr = press_conv_thr

        self.input_text = self.get_input_text()

        # write input file
        self.write_file()

    def get_species_txt(self):
        spectxt = ''
        spectxt += 'ATOMIC_SPECIES'
        for name, mass, pseudo in zip(self.ion_names,
                                      self.ion_masses,
                                      self.ion_pseudo):
            spectxt += '\n {}  {}  {}'.format(name, mass, pseudo)

        return spectxt

    def get_position_txt(self):
        postxt = ''
        postxt += 'ATOMIC_POSITIONS {angstrom}'
        for spec, pos in zip(self.species, self.positions):
            postxt += '\n {} {:1.5f} {:1.5f} {:1.5f}'.format(spec, *pos)

        return postxt

    def get_cell_txt(self):
        celltxt = ''
        celltxt += 'CELL_PARAMETERS {angstrom}'
        for vector in self.cell:
            celltxt += '\n {:1.5f} {:1.5f} {:1.5f}'.format(*vector)

        return celltxt

    def get_kpt_txt(self):
        ktxt = ''
        ktxt += 'K_POINTS automatic'
        ktxt += '\n {} {} {}  0 0 0'.format(*self.kvec)

        return ktxt

    def get_input_text(self):

        input_text = """ &control
    calculation = '{}'
    pseudo_dir = '{}'
    outdir = '{}'
    tstress = .true.
    tprnfor = .true.""".format(self.calculation, self.pseudo_dir, self.outdir)

        # if MD, add time step and number of steps
        if self.calculation == 'md':
            input_text += """
    dt = {}
    nstep = {}""".format(self.dt, self.nstep)

        input_text += """
 /
 &system
    ibrav= 0
    nat= {}
    ntyp= {}
    ecutwfc ={}
    ecutrho = {}""".format(self.nat, self.ntyp, self.ecutwfc,
                           self.ecutrho)

        # if MD or relax, don't reduce number of k points based on symmetry,
        # since the symmetry might change throughout the calculation
        if (self.calculation == 'md') or (self.calculation == 'relax'):
            input_text += """
    nosym = .true."""

        input_text += """
 /
 &electrons
    conv_thr =  1.0d-10
    mixing_beta = 0.7"""

        # if MD or relax, need to add an &IONS block
        if (self.calculation == 'md') or (self.calculation == 'relax') or \
           (self.calculation == 'vc-relax'):
            input_text += """
 /
 &ions"""

        # if MD, add additional details about ions
        if self.calculation == 'md':
            input_text += """
    pot_extrapolation = 'second-order'
    wfc_extrapolation = 'second-order'
    ion_temperature = '{}'
    tempw = {}""".format(self.ion_temperature, self.tempw)

        # if vc-relax, add cell block
        if self.calculation == 'vc-relax':
            input_text += """
 /
 &cell
    press_conv_thr = {}""".format(self.press_conv_thr)

        # insert species, cell, position and k-point textblocks
        input_text += """
 /
{}
{}
{}
{}
""".format(self.species_txt, self.cell_txt, self.position_txt, self.kpt_txt)

        return input_text

    def write_file(self):
        with open(self.input_file_name, 'w') as fin:
            fin.write(self.input_text)

    def run_espresso(self):

        qe_command = 'mpirun {0} < {1} > {2}'.format(self.pw_loc,
                                                     self.input_file_name,
                                                     self.output_file_name)

        os.system(qe_command)


if __name__ == '__main__':
    pass
