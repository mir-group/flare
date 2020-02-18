'''
This module provides OTF training with ASE MD engines: VerlocityVerlet, NVTBerendsen, NPTBerendsen, NPT and Langevin. 
Please see the function `otf_md` below for usage
'''
import os
import sys
from flare.struc import Structure
from flare.ase.otf import OTF

import numpy as np
from ase.calculators.espresso import Espresso
from ase.calculators.eam import EAM
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.md import MolecularDynamics
from ase.md.langevin import Langevin
from ase import units

class OTF_VelocityVerlet(VelocityVerlet, OTF):
    """
    On-the-fly training with ASE's VelocityVerlet molecular dynamics engine.
    Inherit from ASE `VelocityVerlet <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.verlet.VelocityVerlet>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    Args: 
        atoms, timestep, trajectory, dt:
            see `VelocityVerlet <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.verlet.VelocityVerlet>`_
        kwargs: same parameters as :class:`flare.ase.OTF`
    """

    def __init__(self, atoms, timestep=None, trajectory=None, dt=None, 
                 **kwargs):

        VelocityVerlet.__init__(self, atoms, timestep, trajectory, dt=dt)
        OTF.__init__(self, **kwargs)
        
        self.md_engine = 'VelocityVerlet'

class OTF_NVTBerendsen(NVTBerendsen, OTF):
    """
    On-the-fly training with ASE's NVTBerendsen molecular dynamics engine. \
    Inherit from ASE `NVTBerendsen <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.nvtberendsen>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    Args:
        atoms, timestep, temperature, taut, fixcm: see\
            `NVTBerendsen <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.nvtberendsen>`_.
        kwargs: same parameters as :class:`flare.ase.OTF`
    """


    def __init__(self, atoms, timestep, temperature, taut, fixcm=True,
                 trajectory=None, **kwargs):

        NVTBerendsen.__init__(self, atoms, timestep, temperature, taut, 
                              fixcm, trajectory)
 
        OTF.__init__(self, **kwargs)

        self.md_engine = 'NVTBerendsen'

class OTF_NPTBerendsen(NPTBerendsen, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. \
    Inherit from ASE `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    Args:
        atoms, timestep, temperature, taut, pressure, taup, compressibility, fixcm:\
            see `NPTBerendsen <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.nptberendsen>`_.
        kwargs: same parameters as :class:`flare.ase.OTF`
    """


    def __init__(self, atoms, timestep, temperature, taut=0.5e3,
                 pressure=1.01325, taup=1e3,
                 compressibility=4.57e-5, fixcm=True, trajectory=None,
                 **kwargs):

        NPTBerendsen.__init__(self, atoms, timestep, temperature, taut,
                              pressure, taup,
                              compressibility, fixcm, trajectory)

        OTF.__init__(self, **kwargs)

        self.md_engine = 'NPTBerendsen'

class OTF_NPT(NPT, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. \
    Inherit from ASE `NPT <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.npt>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    Args:
        atoms, timestep, temperature, friction:\
            see `NPT <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.npt>`_
        kwargs: same parameters as :class:`flare.ase.OTF`
    """


    def __init__(self, atoms, timestep, temperature, externalstress, 
            ttime, pfactor, mask=None, trajectory=None, **kwargs):

        NPT.__init__(self, atoms, timestep, temperature,
                     externalstress, ttime, pfactor, mask,
                     trajectory)

        OTF.__init__(self, **kwargs)

        self.md_engine = 'NPT'

class OTF_Langevin(Langevin, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. \
    Inherit from ASE `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    Args:
        atoms, timestep, temperature, friction:\
            see `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_.
        kwargs: same parameters as :class:`flare.ase.OTF`
    """

    def __init__(self, atoms, timestep=None, temperature=None, friction=None, 
                 fixcm=True, trajectory=None, **kwargs):

        Langevin.__init__(self, atoms, timestep, temperature, friction, 
                          fixcm, trajectory)

        OTF.__init__(self, **kwargs)
        
        self.md_engine = 'Langevin'



def otf_md(md_engine: str, atoms, md_params: dict, otf_params: dict):
    '''
    Create an OTF MD engine 
    
    Args:
        md_engine (str): the name of md engine, including `VelocityVerlet`,
            `NVTBerendsen`, `NPTBerendsen`, `NPT`, `Langevin`
        atoms (Atoms): ASE Atoms to apply this md engine
        md_params (dict): parameters used in MD engines, 
            must include: `timestep`, `trajectory` (usually set to None).
            Also include those parameters required for ASE MD engine, 
            please look at ASE website to find out parameters for different engines
        otf_params (dict): parameters used in OTF module

    Return:
        An OTF MD class object

    Example:
        >>> from ase import units
        >>> from ase.spacegroup import crystal
        >>> super_cell = crystal(['Ag', 'I'],  
                                 basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                                 size=(2, 1, 1),
                                 cellpar=[3.85, 3.85, 3.85, 90, 90, 90])
        >>> md_engine = 'VelocityVerlet'
        >>> md_params = {'timestep': 1 * units.fs, 'trajectory': None, 
                         'dt': None} 
        >>> otf_params = {'dft_calc': dft_calc, 
                          'init_atoms': [0],
                          'std_tolerance_factor': 1, 
                          'max_atoms_added' : len(super_cell.positions),
                          'freeze_hyps': 10, 
                          'use_mapping': False}
        >>> test_otf = otf_md(md_engine, super_cell, md_params, otf_params)
    '''

    md = md_params
    timestep = md['timestep']
    trajectory = md['trajectory']

    if md_engine == 'VelocityVerlet':
        return OTF_VelocityVerlet(atoms, timestep, trajectory, dt=md['dt'],
                **otf_params)
       
    elif md_engine == 'NVTBerendsen':
        return OTF_NVTBerendsen(atoms, timestep, md['temperature'], 
                md['taut'], md['fixcm'], trajectory, **otf_params)
    
    elif md_engine == 'NPTBerendsen':
        return OTF_NPTBerendsen(atoms, timestep, md['temperature'], 
                md['taut'], md['pressure'], md['taup'], 
                md['compressibility'], md['fixcm'], trajectory, **otf_params)

    elif md_engine == 'NPT':
        return OTF_NPT(atoms, timestep, md['temperature'],
                md['externalstress'], md['ttime'], md['pfactor'], 
                md['mask'], trajectory, **otf_params)

    elif md_engine == 'Langevin':
        return OTF_Langevin(atoms, timestep, md['temperature'],
                md['friction'], md['fixcm'], trajectory, **otf_params)

    else:
        raise NotImplementedError(md_engine+' is not implemented')

