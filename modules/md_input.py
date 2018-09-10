import numpy as np


class MDInput:
    def __init__(self, pseudo_dir: str, outdir: str, dt: float, nstep: int,
                 nat: int, ntyp: int, ecut: float, cell: np.ndarray,
                 species: list, positions: list, nk: np.ndarray,
                 ion_names: list, ion_masses: list, ion_pseudo: list,
                 ion_temperature: str = 'rescaling',
                 tempw: float = 1000):
        self.pseudo_dir = pseudo_dir
        self.outdir = outdir
        self.dt = dt
        self.nstep = nstep
        self.nat = nat
        self.ntyp = ntyp
        self.ecut = ecut
        self.cell = cell
        self.species = species
        self.positions = positions
        self.nk = nk
        self.ion_names = ion_names
        self.ion_masses = ion_masses
        self.ion_pseudo = ion_pseudo
        self.ion_temperature = ion_temperature
        self.tempw = tempw

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

    @staticmethod
    def write_file(fname, text):
        with open(fname, 'w') as fin:
            fin.write(text)

    @staticmethod
    def md_input(pseudo_dir, outdir, dt, nstep,
                 nat, ntyp, ecut, cell, pos, nk,
                 ion_temperature, tempw):

        md_text = """ &control
    calculation = 'md'
    pseudo_dir = '{0}'
    outdir = '{1}'
    dt = {2}
    nstep = {3}
 /
 &system
    ibrav= 0
    nat= {4}
    ntyp= {5}
    ecutwfc ={6}
    nosym = .true.
 /
 &electrons
    conv_thr =  1.0d-10
    mixing_beta = 0.7
 /
 &ions
    pot_extrapolation = 'second-order'
    wfc_extrapolation = 'second-order'
 /
ATOMIC_SPECIES
 Si  28.086  Si.pz-vbc.UPF
 C  12.011  C.pz-rrkjus.UPF
{7}
{8}
K_POINTS automatic
 {9} {9} {9}  0 0 0
    """.format(pseudo_dir, outdir, dt, nstep,
               nat, ntyp, ecut, cell, pos, nk)

        return md_text

if __name__ == '__main__':
    # make test input
    pseudo_dir = 'test/pseudo'
    outdir = '.'
    dt = 20
    nstep = 1000
    nat = 2
    ntyp = 2
    ecut = 18.0
    cell = np.eye(3)
    species = ['C', 'Si']
    positions = [np.array([0, 0, 0]),
                 np.array([0.5, 0.5, 0.5])]
    nk = np.array([4, 4, 4])
    ion_names = ['C', 'Si']
    ion_masses = [2.0, 3.0]
    ion_pseudo = ['pseudo/c', 'pseudo/si']

    test_md = MDInput(pseudo_dir, outdir, dt, nstep, nat, ntyp, ecut,
                      cell, species, positions, nk, ion_names, ion_masses,
                      ion_pseudo)

    test_spec = test_md.get_species_txt()
    print(test_spec)
