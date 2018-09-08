#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to facilitate the 'punch-out' scheme of large-scale GP-accelerated
molecular dynamics simulations.
"""

import numpy as np

from struc import Structure


def is_within_d_box(pos1: np.ndarray,pos2: np.ndarray,d)-> bool:
    """
    Return if pos2 is within a cube of side length d centered around pos1,
    :param pos1:
    :param pos2:
    :param d:
    :return:
    """

    isnear_x = abs(pos1[0] - pos2[0]) <=  d/2
    isnear_y = abs(pos1[1] - pos2[1]) <=  d/2
    isnear_z = abs(pos1[2] - pos2[2]) <=  d/2

    return isnear_x and isnear_y and isnear_z



def punchout(structure: Structure, atom: int, d: float, center : bool = True):
    """

    :param structure: a Structure type object
    """
    a = structure.vec1
    b = structure.vec2
    c = structure.vec3

    shift_vectors = [ np.zeros(3),
        a,b,c,
        -a,-b,-c,
        a+b,a-b,a+c,a-c,
        b+c,b-c,
        -a+b,-a-b,
        -a+c,-a-c,
        -b+c,-b-c,
        a+b+c,
        -a+b+c, a-b+c, a+b-c,
        -a-b+c, -a+b-c, a-b-c,
        -a-b-c
    ]

    new_pos = []
    new_prev_pos = [] # Necessary so new structure has non-zero velocity
    new_species = []

    assert 0 <= atom <= len(structure.positions), 'Atom index  is greater ' \
                                                 'than number of atoms ' \
                                                  'in structure'
    target_pos = structure.positions[atom]


    for i, pos in enumerate(structure.positions):
        # Compute all shifted positions to handle edge cases
        shifted_positions = [pos + shift for shift in shift_vectors]

        for j, shifted_pos in enumerate(shifted_positions):
            if is_within_d_box(target_pos, shifted_pos,d):
                new_pos.append(shifted_pos)
                new_prev_pos.append(structure.prev_positions
                                    [i]+shift_vectors[j])
                new_species.append(structure.species[i])

    # Set up other new structural properties
    newlatt = d*np.eye(3)
    new_mass_dict = {}

    for spec in set(new_species):
        new_mass_dict[spec] = structure.mass_dict[spec]

    # Instantiate new structure, and set the previous positions manually

    newstruc = Structure(newlatt,new_species,new_pos,
                         structure.cutoff,new_mass_dict)
    newstruc.prev_positions = list(new_prev_pos)

    # Center new structure at origin
    if center:
        newstruc.translate_positions(-target_pos)

    return newstruc





#OLD DRAFT METHOD: DEPRECATED

def simple_cube_local_cell(atoms):
    """
    Turns a structure in to a simple cubic cell with 1 angstrom of 'padding' on all sides.
    Used for 'punch-outs'.

    :param structure:
    :return:
    """

    positions = [at.position for at in atoms]

    minx = min([pos[0] for pos in positions])
    miny = min([pos[1] for pos in positions])
    minz = min([pos[2] for pos in positions])

    maxx = max([pos[0] for pos in positions])
    maxy = max([pos[1] for pos in positions])
    maxz = max([pos[2] for pos in positions])

    x_extent = maxx-minx
    y_extent = maxy-miny
    z_extent = maxz-minz

    x_distances=[]
    y_distances=[]
    z_distances=[]

    for i,pos1 in enumerate(positions):
        for j,pos2 in enumerate(positions):
            if pos1[0]-pos2[0]!=0:
                x_distances.append(np.abs(pos1[0]-pos2[0])/2.)
            if pos1[1]-pos2[1]!=0:
                y_distances.append(np.abs(pos1[1]-pos2[1])/2.)
            if pos1[2]-pos2[2]!=0:
                z_distances.append(np.abs(pos1[2]-pos2[2])/2.)


    ####
    # NOT RELEVANT RN:  Give each atom a buffer of 1 angstrom on the side of the cubes
    #for at in atoms:
    #    at.position+=np.array([1,1,1])

    xijbar= np.mean(x_distances)
    yijbar= np.mean(y_distances)
    zijbar= np.mean(z_distances)

    lattice = np.diag([x_extent+xijbar,y_extent+yijbar,z_extent+zijbar])

    return Structure(atoms,alat=1,lattice=lattice)


if __name__=='__main__':



    #from ase.build.bulk import bulk
    #from ase.build.supercells import make_supercell

    #ats = bulk('Si','fcc',a=5.431)

    #Sats = make_supercell(ats,7*np.eye(3))

    #struc=ase_to_structure(Sats,alat=5.14,fractional=False,perturb=.5)

    #print(struc)
    #struc = punchout_cell(struc,target_position=(5.0,5.0,5.0),box_dist=5.0)

    #print(struc)
    """spec ='H'

    positions=[]

    for i in range(5):
        for j in range(5):
            for k in range(5):
                positions.append((i,j,k))
    print(positions)
    print(len(positions))

    alat =1.0
    lattice = 5*np.eye(3)

    atoms = [Atom(position=pos,element='H') for pos in positions]


    struc= Structure(atoms,alat=1,lattice=5*np.eye(3),fractional=False)
    print(struc[2])
    b = get_box_neighbors(struc,target_atom=struc[2],box_dist=.5)
    for x in b:
        print(x)


    c = generate_ultracell(struc)
    for x in c:
        pass
        #print(x)
    print(len(struc))
    print(len(c))

    ultra_pos = [tuple(x.position) for x in c]
    print(len(ultra_pos))
    print(len(set(ultra_pos)))

    print(get_box_neighbors(struc,target_pos=[0,0,0],box_dist=1.0))
    print('')
    nbr_atoms = get_box_neighbors(c,target_pos=[0,0,0],box_dist=1.0)

    print(isinstance(struc,Structure))
    print(simple_cube_local_cell(center_atoms(nbr_atoms)))
    """




















