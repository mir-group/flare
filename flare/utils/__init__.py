"""
Utility functions for various tasks.
"""
from json import JSONEncoder
import numpy as np
from datetime import datetime
from math import ceil


class NumpyEncoder(JSONEncoder):
    """
    Special json encoder for numpy types for serialization
    use as

    json.loads(... cls = NumpyEncoder)

    or:

    json.dumps(... cls = NumpyEncoder)

    Thanks to StackOverflow users karlB and fnunnari, who contributed this from:
    `https://stackoverflow.com/a/47626762`
    """

    def default(self, obj):
        """"""
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def write_embed_coeffs(filename, embed_coeffs, n_species, n_max, l_max, contributor):
    assert embed_coeffs.shape[0] % 2 == 0
    assert embed_coeffs.shape[1] == n_species * n_max * (l_max + 1)

    now = datetime.now()
    d_embed = embed_coeffs.shape[0] // 2

    with open(filename, "w") as f:
        f.write(f"DATE: {now}. CONTRIBUTOR: {contributor}\n")
        f.write(f"{d_embed}\n")
        embed_array = np.reshape(embed_coeffs, -1)
        n_cols = 5
        n_rows = len(embed_array) // n_cols
        n_residual = len(embed_array) % n_cols
        
        # before residual
        coeff_body = np.reshape(embed_array[:n_rows * n_cols], (n_rows, n_cols))
        coeff_str = "\n".join([" ".join([str(l) for l in line]) for line in coeff_body])

        # residual
        coeff_residual = embed_array[-n_residual:]
        coeff_str += "\n" + " ".join([str(l) for l in coeff_residual])

        f.write(coeff_str)
