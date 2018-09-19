import numpy as np


def parse_pos_relax(filename="relaxpos.txt"):
    with open(filename, 'r') as outf:
        lines = outf.readlines()

    pos = np.zeros(shape=(len(lines), 3))

    for index, line in enumerate(lines):
        line = line.split()
        pos[index][0] = str(line[1])
        pos[index][1] = str(line[2])
        pos[index][2] = str(line[3])

    return pos

if __name__ == "__main__":
    print(parse_pos_relax())