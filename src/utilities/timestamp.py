import datetime
import os


def timestamp():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d_%m_%y__%H_%M_%S")
    return formatted_time


def add_timestamp(path):
    folder, full_filename = os.path.split(path)  # Split into folder and filename
    filename, extension = os.path.splitext(
        full_filename
    )  # Split filename and extension
    timestamp_ = timestamp()  # Get current timestamp
    new_path = os.path.join(folder, "__".join([filename, timestamp_]) + extension)
    return new_path  # Return the new path
