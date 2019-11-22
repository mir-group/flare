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

    :param atoms: ASE Atoms object
    :param timestep: time step in units of pico-second
    :param trajectory: the trajectory file dumped to (we usually don't set this because we have our own logger)
    :param dt: usually not set, to be degraded
    :param **kwargs: same parameters as `flare.ase.OTF`
    """

    def __init__(self, atoms, timestep=None, trajectory=None, dt=None, 
                 **kwargs):

        VelocityVerlet.__init__(self, atoms, timestep, trajectory,
                                dt=dt)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)
        
        self.md_engine = 'VelocityVerlet'

class OTF_NVTBerendsen(NVTBerendsen, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. 
    Inherit from ASE `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    :param atoms: ASE Atoms object
    :param timestep: time step in units of pico-second
    :param temperature: temperature
    :param friction: Typical values for the friction are 0.01-0.02 atomic units.
    :param **kwargs: same parameters as `flare.ase.OTF`
    """


    def __init__(self, atoms, timestep, temperature, taut, fixcm=True,
                 trajectory=None, **kwargs):

        NVTBerendsen.__init__(self, atoms, timestep, temperature, taut, 
                              fixcm, trajectory)
 
        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)

        self.md_engine = 'NVTBerendsen'

class OTF_NPTBerendsen(NPTBerendsen, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. 
    Inherit from ASE `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    :param atoms: ASE Atoms object
    :param timestep: time step in units of pico-second
    :param temperature: temperature
    :param friction: Typical values for the friction are 0.01-0.02 atomic units.
    :param **kwargs: same parameters as `flare.ase.OTF`
    """


    def __init__(self, atoms, timestep, temperature, taut=0.5e3,
                 pressure=1.01325, taup=1e3,
                 compressibility=4.57e-5, fixcm=True, trajectory=None,
                 **kwargs):

        NPTBerendsen.__init__(self, atoms, timestep, temperature, taut,
                              pressure, taup,
                              compressibility, fixcm, trajectory)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)

        self.md_engine = 'NPTBerendsen'

class OTF_NPT(NPT, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. 
    Inherit from ASE `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    :param atoms: ASE Atoms object
    :param timestep: time step in units of pico-second
    :param temperature: temperature
    :param friction: Typical values for the friction are 0.01-0.02 atomic units.
    :param **kwargs: same parameters as `flare.ase.OTF`
    """


    def __init__(self, atoms, timestep, temperature, externalstress, 
            ttime, pfactor, mask=None, trajectory=None, **kwargs):

        NPT.__init__(self, atoms, timestep, temperature,
                     externalstress, ttime, pfactor, mask,
                     trajectory)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)

        self.md_engine = 'NPT'

class OTF_Langevin(Langevin, OTF):
    """
    On-the-fly training with ASE's Langevin molecular dynamics engine. 
    Inherit from ASE `Langevin <https://wiki.fysik.dtu.dk/ase/ase/md.html#ase.md.langevin.Langevin>`_ class and our ASE-coupled on-the-fly training engine `flare.ase.OTF`

    :param atoms: ASE Atoms object
    :param timestep: time step in units of pico-second
    :param temperature: temperature
    :param friction: Typical values for the friction are 0.01-0.02 atomic units.
    :param **kwargs: same parameters as `flare.ase.OTF`
    """

    def __init__(self, atoms, timestep=None, temperature=None, friction=None, 
                 trajectory=None, **kwargs):

        Langevin.__init__(self, atoms, timestep, temperature, friction)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)
        
        self.md_engine = 'Langevin'



def otf_md(md_engine: str, atoms, md_params: dict, otf_params: dict):

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
                md['friction'], trajectory, **otf_params)

    else:
        raise NotImplementedError(md_engine+' is not implemented')

