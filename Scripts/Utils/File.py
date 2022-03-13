import os
import errno

def check_path(filepath):

    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def get_file_path(paths):
    return os.path.sep.join(paths)

def get_file_name(path):
    path_splited = path.split(os.path.sep)

    return path_splited[-1]

def files_exists(files):
    for f in files:
        if not os.path.isfile(f):
            return False

    return True