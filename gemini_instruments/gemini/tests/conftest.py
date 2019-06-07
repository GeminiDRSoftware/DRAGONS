
import glob
import pytest
import os


@pytest.fixture(scope="module")
def gemini_files():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    def get_files(instrument):
        return glob.glob(os.path.join(path, instrument, "*fits"))

    gemini_files = []
    gemini_files.extend(get_files("GMOS"))
    gemini_files.extend(get_files("F2"))

    yield gemini_files
