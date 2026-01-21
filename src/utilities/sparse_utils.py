import scipy.sparse as sp
import numpy as np
from scipy.sparse import vstack

def indexing_csr(csr_array, idx_np_array, chunksize):
    # https://stackoverflow.com/a/20344429
    list_chunks = np.array_split(idx_np_array, chunksize)
    chunks = [csr_array[chunk] 
            for chunk in list_chunks]
    result = vstack(chunks)
    return result

def convert_to_32_bit(A, *, copy=True, check_overflow=True):
    """
    Convert ONLY the value dtype (via SciPy sparse .astype), keeping the sparse
    format and index arrays unchanged.

    - float64 -> float32
    - int64   -> int32

    Parameters
    ----------
    A : scipy.sparse.spmatrix
    copy : bool
        Passed through to scipy.sparse .astype(copy=copy).
    check_overflow : bool
        For int64 -> int32, optionally raise if values won't fit in int32.

    Returns
    -------
    scipy.sparse.spmatrix
    """
    if not sp.isspmatrix(A):
        raise TypeError("A must be a SciPy sparse matrix")

    dt = A.dtype

    if np.issubdtype(dt, np.floating):
        if dt == np.float64:
            return A.astype(np.float32, copy=copy)
        return A.copy() if copy else A

    if np.issubdtype(dt, np.integer):
        if dt == np.int64:
            if check_overflow and A.nnz:
                info = np.iinfo(np.int32)
                dmin = A.data.min()
                dmax = A.data.max()
                if dmin < info.min or dmax > info.max:
                    raise OverflowError("Sparse data values do not fit in int32")
            return A.astype(np.int32, copy=copy)
        return A.copy() if copy else A

    raise TypeError(f"Unsupported dtype {dt}. Expected int or float sparse matrix.")
