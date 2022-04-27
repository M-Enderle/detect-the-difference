from glob import glob
import json

def list_files(filename):
    """ List file names """

    return sorted(glob(filename))


def load_results(results_file):
    """ Load results from file """

    with open(results_file, 'r') as file_obj:
        return json.load(file_obj)
