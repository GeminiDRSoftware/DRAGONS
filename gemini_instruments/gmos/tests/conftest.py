import glob
import os

import pytest

from astrodata.test import conftest

input_test_path = conftest.input_test_path


@pytest.fixture
def gmos_files(input_test_path):
    def get_files(instrument):
        return glob.glob(os.path.join(input_test_path, instrument, "*fits"))

    gemini_files = []
    gemini_files.extend(get_files("GMOS"))

    yield gemini_files
