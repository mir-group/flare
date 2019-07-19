import os
import sys
sys.path.append('../..')
from flare.struc import Structure
from flare.modules.ase_otf import OTF

import numpy as np
from ase.calculators.espresso import Espresso
from ase.calculators.eam import EAM
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.md import MolecularDynamics
from ase import units


class OTF_VelocityVerlet(VelocityVerlet, OTF):
    def __init__(self, atoms, timestep=None, trajectory=None, dt=None, 
                 **kwargs):

        VelocityVerlet.__init__(self, atoms, timestep, trajectory,
                                dt=dt)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)


class OTF_NVTBerendsen(NVTBerendsen, OTF):
    def __init__(self, atoms, timestep, temperature, taut, fixcm=True,
                 trajectory=None, **kwargs):

        NVTBerendsen.__init__(self, atoms, timestep, temperature, taut, 
                              fixcm, trajectory)
 
        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)


class OTF_NPTBerendsen(NPTBerendsen, OTF):
    def __init__(self, atoms, timestep, temperature, taut=0.5e3 *
                 units.fs, pressure=1.01325, taup=1e3 * units.fs,
                 compressibility=4.57e-5, fixcm=True, trajectory=None,
                 **kwargs):

        NPTBerendsen.__init__(self, atoms, timestep, temperature, taut,
                              pressure, taup,
                              compressibility, fixcm, trajectory)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)


class OTF_NPT(NPT, OTF):
    def __init__(self, atoms, timestep, temperature, externalstress, 
            ttime, pfactor, mask=None, trajectory=None, **kwargs):

        NPT.__init__(self, atoms, timestep, temperature,
                     externalstress, ttime, pfactor, mask,
                     trajectory)

        OTF.__init__(self, atoms, timestep, trajectory, **kwargs)


