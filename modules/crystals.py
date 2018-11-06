import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.io import write


def get_supercell_positions(sc_size, cell, positions):
    sc_positions = []
    for m in range(sc_size):
        vec1 = m * cell[0]
        for n in range(sc_size):
            vec2 = n * cell[1]
            for p in range(sc_size):
                vec3 = p * cell[2]

                # append translated positions
                for pos in positions:
                    sc_positions.append(pos+vec1+vec2+vec3)

    return sc_positions


# -----------------------------------------------------------------------------
#                          fcc helper functions
# -----------------------------------------------------------------------------

def fcc_positions(cube_lat):
    positions = [np.array([0, 0, 0]),
                 np.array([cube_lat/2, cube_lat/2, 0]),
                 np.array([0, cube_lat/2, cube_lat/2]),
                 np.array([cube_lat/2, 0, cube_lat/2])]
    return positions

# -----------------------------------------------------------------------------
#                        diamond helper functions
# -----------------------------------------------------------------------------


def cubic_diamond_positions(cube_lat):
    positions = [np.array([0, 0, 0]),
                 np.array([cube_lat/2, cube_lat/2, 0]),
                 np.array([0, cube_lat/2, cube_lat/2]),
                 np.array([cube_lat/2, 0, cube_lat/2]),
                 np.array([cube_lat/4, cube_lat/4, cube_lat/4]),
                 np.array([3*cube_lat/4, 3*cube_lat/4, cube_lat/4]),
                 np.array([cube_lat/4, 3*cube_lat/4, 3*cube_lat/4]),
                 np.array([3*cube_lat/4, cube_lat/4, 3*cube_lat/4])]
    return positions


def primitive_diamond_positions(prim_lat):
    positions = [np.array([0, 0, 0]),
                 np.array([prim_lat/2, prim_lat/2, prim_lat/2])]
    return positions


# -----------------------------------------------------------------------------
#                        slab helper functions
# -----------------------------------------------------------------------------

def get_fcc111_slab(layers, size, element, vacuum):
    slab = fcc111(element, size=(size, size, layers), vacuum=vacuum)
    return slab


def fcc111_and_adsorbate(layers, size, element, vacuum, height, position):
    slab = fcc111(element, size=(size, size, layers))
    add_adsorbate(slab, element, height, position)
    slab.center(vacuum=vacuum, axis=2)
    return slab


# -----------------------------------------------------------------------------
#                        water helper functions
# -----------------------------------------------------------------------------


def water_coordinates(ox_pos: np.ndarray,
                      theta: float, phi: float) -> list:
    H_angle = 104.45 * (2*np.pi / 360)
    OH_len = 95.84e-12
    pass


if __name__ == '__main__':
    layers = 2
    size = 2
    element = 'Pd'
    vacuum = 10
    height = 1
    position = 'hcp'

    slab_test = fcc111_and_adsorbate(layers, size, element, vacuum, height,
                                     position)
    print(slab_test.positions)
    print(slab_test.cell)
