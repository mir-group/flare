"""
Utility functions for various tasks.
"""
from json import JSONEncoder
import numpy as np


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
