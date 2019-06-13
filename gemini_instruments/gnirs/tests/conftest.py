import glob
import os

import pytest

from astrodata.test.conftest import input_test_path


@pytest.fixture
def gemini_files(input_test_path):

    def get_files(instrument):
        return glob.glob(os.path.join(input_test_path, instrument, "*fits"))

    gemini_files = []
    gemini_files.extend(get_files("Archive"))

    yield gemini_files
