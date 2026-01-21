import traceback
import warnings
import sys
import numpy as np
import traceback
import sys
import os
from pathlib import Path

# TO USE WITH warnings.showwarning = warn_with_traceback

# # Custom function to format and print the traceback
# def custom_excepthook(exc_type, exc_value, exc_traceback):
#     # Print the exception with clickable file:line format
#     formatted_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
#     sys.stderr.write(formatted_traceback)

# # Set the custom excepthook
# sys.excepthook = custom_excepthook


def filter_stack_trace(stack):
    cutoff_index = None
    for i, frame in enumerate(reversed(stack)):
        if "IPython/core/interactiveshell.py" in frame.filename:
            cutoff_index = np.flip(np.arange(len(stack)))[i] + 1
            break

    # If we found the frame, slice the stack to include only frames up to that point
    if cutoff_index is not None:
        stack = stack[cutoff_index:]

    # Get the current working directory (assumed to be the root of the user's project)
    project_path = os.getcwd()

    # Get system paths to exclude (site-packages, standard Python libs)
    system_paths = [sys.prefix, sys.exec_prefix, os.path.dirname(os.__file__)]

    # Filter out system libraries and only include frames from your project and imported third-party libraries, remove this file..
    filtered_stack = [
        frame
        for frame in stack
        if (
            frame.filename.startswith(project_path)
            or not any(frame.filename.startswith(path) for path in system_paths)
        )
        and (frame.filename != __file__)
    ]

    return filtered_stack


def clickable_traceback():
    # Get the formatted stack trace as a list of strings
    stack_trace = filter_stack_trace(traceback.extract_stack())

    # Loop through the stack trace and format each entry
    formatted_traceback = ""
    for frame in stack_trace:
        # Each frame is a tuple: (filename, lineno, function, text)
        filename, lineno, func, text = frame
        formatted_traceback += f"{filename}:{lineno} in {func} -> {text}\n"

    # Now, you can do something with the formatted trace, like printing it or logging it
    return formatted_traceback


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    log.write(clickable_traceback())
    # log.write(warnings.formatwarning(message, category, filename, lineno, line))
    log.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")
    log.write("\n")


def default_warning(message, category, filename, lineno, file=None, line=None):
    # default python warning
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))


def strip_up_to_dir(path: Path, anchor_dir: str) -> Path:
    parts = path.parts
    try:
        idx = parts.index(anchor_dir)
        return Path(*parts[idx + 1 :])  # skip 'bin2cell' itself
    except ValueError:
        raise ValueError(f"'{anchor_dir}' not found in path: {path}")


import warnings
from contextlib import contextmanager


@contextmanager
def warn_with_prefix(prefix: str):
    with warnings.catch_warnings():
        warnings.simplefilter("always")

        original_showwarning = warnings.showwarning

        def custom_showwarning(
            message, category, filename, lineno, file=None, line=None
        ):
            original_showwarning(
                f"{prefix}{message}", category, filename, lineno, file, line
            )

        warnings.showwarning = custom_showwarning
        try:
            yield
        finally:
            warnings.showwarning = original_showwarning
