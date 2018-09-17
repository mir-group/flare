import numpy as np
import os


class QEInput:
    def __init__(self, input_file_name: str, output_file_name: str,
                 pw_loc: str, calculation: str,
                 scf_inputs: dict, md_inputs: dict = None):

        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.pw_loc = pw_loc
        self.calculation = calculation

        self.pseudo_dir = scf_inputs['pseudo_dir']
        self.outdir = scf_inputs['outdir']
        self.nat = scf_inputs['nat']
        self.ntyp = scf_inputs['ntyp']
        self.ecutwfc = scf_inputs['ecutwfc']
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

        if self.calculation == 'md':
            self.dt = md_inputs['dt']
            self.nstep = md_inputs['nstep']
            self.ion_temperature = md_inputs['ion_temperature']
            self.tempw = md_inputs['tempw']

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
    tprnfor = .true.""".format(self.calculation, self.pseudo_dir, self.outdir)

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
    nosym = .true.
 /
 &electrons
    conv_thr =  1.0d-10
    mixing_beta = 0.7""".format(self.nat, self.ntyp, self.ecutwfc)

        if (self.calculation == 'md') or (self.calculation == 'relax'):
            input_text += """
 /
 &ions
    pot_extrapolation = 'second-order'
    wfc_extrapolation = 'second-order'"""

        if (self.calculation == 'md'):
            input_text += """
    ion_temperature = '{}'
    tempw = {}""".format(self.ion_temperature, self.tempw)

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

        qe_command = '{0} < {1} > {2}'.format(self.pw_loc,
                                              self.input_file_name,
                                              self.output_file_name)

        os.system(qe_command)

if __name__ == '__main__':
    pass
