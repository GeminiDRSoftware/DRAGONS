
import glob
import pytest
import os


@pytest.fixture(scope="module")
def test_path():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    return path


@pytest.fixture(scope="module")
def archive_files(test_path):

    yield glob.glob(os.path.join(test_path, "Archive/", "*fits"))
