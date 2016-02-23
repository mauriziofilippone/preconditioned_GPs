import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import methods
import preconditioners
import kernels


def load(file_path):
    """
    Load a previously pickled model, using `m.pickle('path/to/file.pickle)'

    :param file_name: path/to/file.pickle
    """
    import cPickle as pickle
    try:
        with open(file_path, 'rb') as f:
            m = pickle.load(f)
    except:
        import pickle as pickle
        with open(file_path, 'rb') as f:
            m = pickle.load(f)
    return m
