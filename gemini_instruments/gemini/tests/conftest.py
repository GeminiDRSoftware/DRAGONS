import glob
import os

import pytest


@pytest.fixture
def gemini_files(path_to_inputs):

    def get_files(instrument):
        return glob.glob(os.path.join(path_to_inputs, instrument, "*fits"))

    gemini_files = []
    gemini_files.extend(get_files("Archive"))
    gemini_files.sort()

    yield gemini_files
