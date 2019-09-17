#!/usr/bin/env python
"""
Script used to download test files from the GEMINI archive and save them inside
the path stored in the TEST_PATH environment variable. Before running it, make
sure that you set this path using the following command:

    $ export TEST_PATH="/path/to/my/test/data/"
    $ echo $TEST_PATH      
      /path/to/my/test/data/

The test data is listed inside the global variable FILE_WITH_TEST_FILES. Each
row has one file. Each file can be preceeded with a subfolder. This is useful
to isolate tests.
"""

import os
import subprocess
import sys


URL = u'https://archive.gemini.edu/file/'


def main():
    """ Main function to download test inputs. """

    args = _parse_arguments()
    path = _get_dragons_input_test_path()
    create_test_folder_if_does_not_exist(path)
    download_non_existing_test_files(path, args.list_of_files)


def _parse_arguments():
    """
    Parses arguments from the command line.

    Returns
    -------
    Namespace : object that contains input command line parameters.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'list_of_files', type=str,
        help='Input ASCII file with a list of files to be downloaded')

    return parser.parse_args()


def _get_dragons_input_test_path():
    """
    Extracts value from the environment variable $DRAGONS_TEST_INPUTS. This
    variable should contain the path to where the data will be stored.

    Returns
    -------
    str : Path where the downloaded data will be stored.

    """
    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_INPUTS'])

    except KeyError:

        print('\n This script needs the environment variable TEST_PATH'
              '\n Please, add is using the following command: '
              '\n     $ export DRAGONS_TEST_INPUTS="/my/test/path/"'
              '\n and run again. Leaving now.'
              '\n ')

        sys.exit(1)

    return path


def create_test_folder_if_does_not_exist(path):
    """
    Checks if path exists. If not, creates it.

    Parameters
    ----------
    path : str
        Path to where the data will be stored
    """
    print('')
    if os.path.exists(path):
        print(' Skip creation of existing folder: {}'.format(path))
    else:
        print(' Create non-existing test folder: {}'.format(path))
        os.makedirs(path)


def download_non_existing_test_files(path, list_of_files):
    """
    Uses curl to download the FITS files listed inside `.jenkins/test_files.txt`
    from the Gemini Archive to be used in local tests or with Jenkins.

    Parameters
    ----------
    path : str
        This is where all the data will be stored

    list_of_files : list
        Names of the files that will be downloaded. Should contain subdirectories.
    """
    with open(list_of_files, 'r') as _files:

        print('')

        for _filename in _files.readlines():

            current_file = os.path.join(path, _filename).strip()

            if len(_filename.strip()) == 0:
                print('')
                continue

            if _filename.startswith('#'):
                print(" {}".format(_filename.strip()))
                continue

            if os.path.exists(current_file):
                print(' Skip existing file: {:s}'.format(current_file))

            else:
                print(' Download missing file: {:s}'.format(current_file))
                _path, _file = os.path.split(current_file)

                if not os.path.exists(_path):
                    os.makedirs(_path)

                try:
                    subprocess.run(['curl', '--silent', URL + _file, '--output',
                                    current_file], check=True)

                except subprocess.CalledProcessError:
                    print(' Failed to download file: {}'.format(current_file))

        print('')


if __name__ == "__main__":
    main()
