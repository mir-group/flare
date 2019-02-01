import numpy as np
import math
from copy import copy
from typing import List
import subprocess


def write_cfgs_from_md(md_trajectory, start_frame, no_frames, folder_name,
                       image_quality, scr_dest, trans_vec, skip=1):
    # make folder for cfgs
    subprocess.call('mkdir %s' % folder_name, shell=True)
    subprocess.call('mkdir %s/Pic' % folder_name, shell=True)

    scr_anim_text = '%s\n' % image_quality

    # make cfgs
    no_digits = int(np.ceil(math.log(no_frames, 10)))
    cell = md_trajectory.cell
    for n in range(no_frames):
        frame_no = n
        frame_no_padded = str(frame_no).zfill(no_digits)
        frame_string = frame_no_padded+'.cfg'
        frame_dest = folder_name+'/'+frame_string

        scr_anim_text += '%s %s/Pic/%s.jpg\n' % \
            (frame_dest, folder_name, frame_no_padded)

        positions = \
            np.array(md_trajectory.MD_data[start_frame+n*skip]
                     ['positions']) + \
            trans_vec
        species = md_trajectory.MD_data[start_frame+n*skip]['elements']

        write_cfg_file(frame_dest, positions, species, cell)

    # write animation directions for AtomEye
    write_file(scr_dest, scr_anim_text)


def write_cfg_file(file_name: str, positions: np.ndarray, species: List[str],
                   cell: np.ndarray) -> None:
    """write cfg file that can be interpreted by AtomEye.
    assumes orthorombic unit cell.

    :param file_name: destination of cfg file
    :type file_name: str
    :param positions: positions of atoms
    :type positions: np.ndarray
    :param species: atom species
    :type species: List[str]
    :param cell: unit cell vectors
    :type cell: np.ndarray
    :return: creates the cfg file
    :rtype: None
    """

    cfg_text = get_cfg_text(positions, species, cell)
    write_file(file_name, cfg_text)


def get_cfg_text(positions: np.ndarray, species: List[str],
                 cell: np.ndarray) -> str:
    """returns cfg text

    :param positions: Cartesian coordinates of atomic positions
    :type positions: np.ndarray
    :param species: list of atomic species (determines atom size)
    :param cell: cell of unit vectors
    :type cell: np.ndarray
    :return: cfg text
    :rtype: str
    """

    cfg_header = get_cfg_header(positions.shape[0], cell)
    reduced_coordinates = calculate_reduced_coordinates(positions, cell)
    position_text = get_reduced_coordinate_text(reduced_coordinates, species)
    cfg_text = cfg_header + position_text

    return cfg_text


def get_cfg_header(number_of_particles: int, cell: np.ndarray) -> str:
    """creates cfg header from atom positions and unit cell.
    assumes unit cell is orthorombic.

    :param positions: Nx3 array of atomic positions
    :type positions: np.ndarray
    :param cell: 3x3 array of cell vectors (cell vectors are rows)
    :type cell: np.ndarray
    """

    cfg_text = """Number of particles = %i
# (required) this must be the first line

A = 1.0 Angstrom (basic length-scale)
# (optional) basic length-scale: default A = 1.0 [Angstrom]

H0(1,1) = %f A
H0(1,2) = 0 A
H0(1,3) = 0 A
# (required) this is the supercell's 1st edge, in A

H0(2,1) = 0 A
H0(2,2) = %f A
H0(2,3) = 0 A
# (required) this is the supercell's 2nd edge, in A

H0(3,1) = 0 A
H0(3,2) = 0 A
H0(3,3) = %f A
# (required) this is the supercell's 3rd edge, in A

Transform(1,1) = 1
Transform(1,2) = 0
Transform(1,3) = 0
Transform(2,1) = 0
Transform(2,2) = 1
Transform(2,3) = 0
Transform(3,1) = 0
Transform(3,2) = 0
Transform(3,3) = 1
# (optional) apply additional transformation on H0:  H = H0 * Transform;
# default = Identity matrix.

eta(1,1) = 0
eta(1,2) = 0
eta(1,3) = 0
eta(2,2) = 0
eta(2,3) = 0
eta(3,3) = 0
# (optional) apply additional Lagrangian strain on H0:
# H = H0 * sqrt(Identity_matrix + 2 * eta);
# default = zero matrix.

# ENSUING ARE THE ATOMS, EACH ATOM DESCRIBED BY A ROW
# 1st entry is atomic mass in a.m.u.
# 2nd entry is the chemical symbol (max 2 chars)

# 3rd entry is reduced coordinate s1 (dimensionless)
# 4th entry is reduced coordinate s2 (dimensionless)
# 5th entry is reduced coordinate s3 (dimensionless)
# real coordinates x = s * H,  x, s are 1x3 row vectors

# 6th entry is d(s1)/dt in basic rate-scale R
# 7th entry is d(s2)/dt in basic rate-scale R
# 8th entry is d(s3)/dt in basic rate-scale R
R = 1.0 [ns^-1]
# (optional) basic rate-scale: default R = 1.0 [ns^-1]
""" % (number_of_particles, cell[0, 0], cell[1, 1], cell[2, 2])

    return cfg_text


def calculate_reduced_coordinates(positions: np.ndarray,
                                  cell: np.ndarray) -> np.ndarray:
    """convert raw cartesian coordinates to reduced coordinates with each atom
    wrapped back into the unit cell. assumes unit cell is orthorombic.

    :param positions: Nx3 array of atomic positions
    :type positions: np.ndarray
    :param cell: 3x3 array of cell vectors (cell vectors are rows)
    :type cell: np.ndarray
    :return: Nx3 array of reduced coordinates
    :rtype: np.ndarray
    """

    reduced_coordinates = np.zeros((positions.shape[0], 3))
    for m in range(positions.shape[0]):
        for n in range(3):
            trial_coord = positions[m, n] / cell[n, n]

            # reduced coordinates must be between 0 and 1
            trans = np.floor(trial_coord)
            reduced_coordinates[m, n] = trial_coord - trans

    return reduced_coordinates


def get_reduced_coordinate_text(reduced_coordinates: np.ndarray,
                                species: List[str]) -> str:
    """records reduced coordinates in cfg format.

    :param reduced_coordinates: array of reduced coordinates
    :type reduced_coordinates: np.ndarray
    :param species: list of atomic species, which determines atom size
    :type species: List[str]
    :return: cfg string of reduced coordinates.
    :rtype: str
    """

    reduced_text = """
# ENSUING ARE THE ATOMS, EACH ATOM DESCRIBED BY A ROW
# 1st entry is atomic mass in a.m.u.
# 2nd entry is the chemical symbol (max 2 chars)

# 3rd entry is reduced coordinate s1 (dimensionless)
# 4th entry is reduced coordinate s2 (dimensionless)
# 5th entry is reduced coordinate s3 (dimensionless)
# real coordinates x = s * H,  x, s are 1x3 row vectors

# 6th entry is d(s1)/dt in basic rate-scale R
# 7th entry is d(s2)/dt in basic rate-scale R
# 8th entry is d(s3)/dt in basic rate-scale R
R = 1.0 [ns^-1]
# (optional) basic rate-scale: default R = 1.0 [ns^-1]
"""

    for spec, coord in zip(species, reduced_coordinates):
        # use arbitrary mass, label
        reduced_text += \
            '1.0 %s %f %f %f 0 0 0 \n' % (spec, coord[0], coord[1], coord[2])

    return reduced_text


def write_file(file_name: str, text: str):
    with open(file_name, 'w') as fin:
        fin.write(text)


if __name__ == '__main__':
    reduced_coordinates = np.array([[1, 2, 3], [4, 5, 6]])
    species = ['A', 'B']
    test = get_reduced_coordinate_text(reduced_coordinates, species)
    print(test)
