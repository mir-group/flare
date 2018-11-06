import numpy as np


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
