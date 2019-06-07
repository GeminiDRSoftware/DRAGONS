
import glob
import pytest
import os


@pytest.fixture(scope="module")
def gmos_files():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    yield glob.glob(os.path.join(path, "GMOS/", "*fits"))
