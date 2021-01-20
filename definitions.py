import os
import pickle

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = ROOT_DIR + '/data'
PUNCTUATION_DIR = ROOT_DIR + '/data/processed/Periodic/Punctuation'


def mkdir_if_not_exists(loc, file=False):
    """Makes a directory if it doesn't already exist. If loc is specified as a file, ensure the file=True option is set.

    Args:
        loc (str): The file/folder for which the folder location needs to be created.
        file (bool): Set true if supplying a file (then will get the dirstring and make the dir).

    Returns:
        None
    """
    loc_ = os.path.dirname(loc) if file else loc
    if not os.path.exists(loc):
        os.makedirs(loc_, exist_ok=True)


def save_pickle(obj, filename, protocol=4, create_folder=True):
    """ Basic pickle/dill dumping.

    Given a python object and a filename, the method will save the object under that filename.

    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.

    Returns:
        None
    """
    if create_folder:
        mkdir_if_not_exists(filename, file=True)

    # Save
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)


def load_pickle(filename):
    """ Basic dill/pickle load function.

    Args:
        filename (str): Location of the object.

    Returns:
        python object: The loaded object.
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj
