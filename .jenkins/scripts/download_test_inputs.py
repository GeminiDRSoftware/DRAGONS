#!/usr/bin/env python
"""
Script used to download test files from the GEMINI archive and save them inside
the path stored in the DRAGONS_TEST_INPUTS environment variable. Before running
it, make sure that you set this path using the following command:

    $ export DRAGONS_TEST_INPUTS="/path/to/my/test/data/"
    $ echo $DRAGONS_TEST_INPUTS
      /path/to/my/test/data/

The test data is listed inside a text file that is passed passed via command
line. Each row has one file. Each file can be preceeded with a subfolder. This
is useful to isolate tests.
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

        print('\n This script needs the environment variable DRAGONS_TEST_INPUTS'
              '\n Please, add is using the following command: '
              '\n     $ export DRAGONS_TEST_INPUTS="/my/test/path/"'
              '\n and run again. Leaving now.'
              '\n ')

        sys.exit(0)

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
        os.makedirs(path, mode=0o775)


def download_non_existing_test_files(path, list_of_files):
    """
    Uses curl to download the FITS files listed inside `.jenkins/test_files.txt`
    from the Gemini Archive to be used in local tests or with Jenkins.

    Parameters
    ----------
    path : str
        This is where all the data will be stored

    list_of_files : str
        Names of the files that will be downloaded. Should contain subdirectories.
    """

    with open(list_of_files, 'r') as _files:

        print('')

        for _filename in _files.readlines():

            # Remove spaces and special characters from string
            current_file = os.path.join(path, _filename).strip()
            current_file = current_file.replace("\x1b", "")

            if len(_filename.strip()) == 0:
                print('')
                continue

            if _filename.startswith('#'):
                print(" {}".format(_filename.strip()))
                continue

            if '#' in current_file:
                current_file = current_file.split('#')[0]
                current_file = current_file.strip()

            if os.path.exists(current_file):
                print(' Skip existing file: {:s}'.format(current_file))

            else:
                print(' Download missing file: {:s}'.format(current_file))
                _path, _file = os.path.split(current_file)

                if not os.path.exists(_path):
                    oldmask = os.umask(000)
                    os.makedirs(_path, mode=0o775)
                    os.umask(oldmask)

                try:
                    subprocess.run(['curl', '--silent', URL + _file, '--output',
                                    current_file], check=True)
                    os.chmod(current_file, mode=0o775)

                except subprocess.CalledProcessError as e:
                    print(' Failed to download file: {}'.format(current_file))

                    if hasattr(e, 'message'):
                        print(e.message)
                    else:
                        print(e)

        print('')


if __name__ == "__main__":
    main()
