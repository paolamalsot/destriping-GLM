import warnings
from contextlib import contextmanager


@contextmanager
def preserve_warnings():
    saved_filters = warnings.filters.copy()
    saved_showwarning = warnings.showwarning
    saved_formatwarning = warnings.formatwarning

    try:
        yield
    finally:
        warnings.filters = saved_filters
        warnings.showwarning = saved_showwarning
        warnings.formatwarning = saved_formatwarning
