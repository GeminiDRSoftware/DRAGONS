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

FILES_TO_BE_CACHED = 'test_files.txt'

URL = u'https://archive.gemini.edu/file/'

try:
    DRAGONS_INPUT_TEST_PATH = os.environ['DRAGONS_TEST_INPUTS']

except KeyError as err:

    print('\n This script needs the environment variable TEST_PATH'
          '\n Please, add is using the following command: '
          '\n     $ export DRAGONS_TEST_INPUTS="/my/test/path/"'
          '\n and run again. Leaving now.'
          '\n ')

    sys.exit(1)


def main():
    create_test_folder_if_does_not_exist()
    download_non_existing_test_files()


def create_test_folder_if_does_not_exist():
    """
    Checks if DRAGONS_INPUT_TEST_PATH exists. If not, creates it.
    """
    print('')
    if os.path.exists(DRAGONS_INPUT_TEST_PATH):
        print(' Skip creation of existing folder: {}'.format(
            DRAGONS_INPUT_TEST_PATH))
    else:
        print(' Create non-existing test folder: {}'.format(
            DRAGONS_INPUT_TEST_PATH))
        os.makedirs(DRAGONS_INPUT_TEST_PATH)


def download_non_existing_test_files():
    """
    Uses curl to download the FITS files listed inside `.jenkins/test_files.txt`
    from the Gemini Archive to be used in local tests or with Jenkins.
    """
    here = os.path.dirname(__file__)
    relative_path = "../"

    full_path_filename = os.path.join(here, relative_path, FILES_TO_BE_CACHED)

    with open(full_path_filename, 'r') as list_of_files:

        print('')

        for _filename in list_of_files.readlines():

            current_file = os.path.join(
                DRAGONS_INPUT_TEST_PATH, _filename).strip()

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
