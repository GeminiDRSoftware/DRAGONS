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


FILE_WITH_TEST_FILES = 'test_files.txt'
URL = u'https://archive.gemini.edu/file/'


try:
    TEST_PATH = os.environ['TEST_PATH']

except KeyError as err:

    print('\n This script needs the environment variable TEST_PATH'
          '\n Please, add is using the following command: ' \
          '\n     $ export TEST_PATH="/my/test/path/"'
          '\n and run again. Leaving now.'
          '\n ')

    sys.exit(1)


def main():

    create_test_folder_if_does_not_exist()
    download_non_existing_test_files()


def create_test_folder_if_does_not_exist():

    print('')
    if os.path.exists(TEST_PATH):
        print(' Skip creation of existing folder: {}'.format(TEST_PATH))
    else:
        print(' Create non-existing test folder: {}'.format(TEST_PATH))
        os.makedirs(TEST_PATH)


def download_non_existing_test_files():

    full_path_filename = os.path.join(
        os.path.dirname(__file__),
        '../',
        FILE_WITH_TEST_FILES
    )

    with open(full_path_filename, 'r') as list_of_files:

        print('')

        for _filename in list_of_files.readlines():

            current_file = os.path.join(TEST_PATH, _filename).strip()

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
