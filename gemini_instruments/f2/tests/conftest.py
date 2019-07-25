import glob
import os

import pytest

from astrodata.testing import path_to_inputs


@pytest.fixture
def f2_files(path_to_inputs):
    def get_files(instrument):
        return glob.glob(os.path.join(path_to_inputs, instrument, "*fits"))

    gemini_files = []
    gemini_files.extend(get_files("F2"))
    gemini_files.sort()

    yield gemini_files
