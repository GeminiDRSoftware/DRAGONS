#!/usr/bin/env bash

import os
import requests
import shutil


FILE_WITH_TEST_FILES = '.jenkins/test_files.txt'
TEST_PATH = os.environ['TEST_PATH']
URL = 'https://archive.gemini.edu/file/{:s}'


def download_test_data():

    create_test_folder_if_does_not_exist()
    download_non_existing_test_files()


def create_test_folder_if_does_not_exist():

    if os.path.exists(TEST_PATH):
        print('Skip creation of existing folder: {}'.format(TEST_PATH))
    else:
        print('Create non-existing test folder: {}'.format(TEST_PATH))
        os.makedirs(TEST_PATH)


def download_non_existing_test_files():

    with open(FILE_WITH_TEST_FILES, 'r') as list_of_files:

        for _filename in list_of_files.readlines():

            current_file = os.path.join(TEST_PATH, _filename)

            if os.path.exists(current_file):
                print('Skip existing file: {}'.format(current_file))

            else:

                print('Download missing file: {}'.format(current_file))
                _path, _file = os.path.split(current_file)
                os.makedirs(_path, exist_ok=True)
                r = requests.get(URL.format(_file), stream=True)

                if r.status_code == 200:
                    with open(current_file, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)


if __name__ == "__main__":
    download_test_data()
