import numpy as np

def get_max_value(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).max
    else:
        raise TypeError("Unsupported dtype")